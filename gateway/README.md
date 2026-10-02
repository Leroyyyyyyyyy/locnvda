# 阶段 4a / 4b：异步 API 代理与鉴权

已实现 JSON / SSE 代理、调用方 API key 鉴权和上游凭证隔离。**还没有限流、在途并发保护或请求体大小限制，继续只绑定本机，不要直接暴露到公网。**

```text
客户端（客户端 key） → 网关 :8080（认证 → 调用方 ID） → vLLM :8000（上游 key）
```

支持：

- `GET /v1/models`：需要客户端 key。
- `POST /v1/completions`：需要客户端 key。
- `POST /v1/chat/completions`：需要客户端 key，包括 SSE 流式请求。
- `GET /health`：无需 key，仅检查网关进程存活，不检查上游/GPU。

## 1. 本地安装（无需 GPU）

在仓库根目录运行，用独立虚拟环境，避免更改 vLLM 的依赖：

```bash
uv venv --python 3.12 .venv-gateway
uv pip install --python .venv-gateway/bin/python -r gateway/requirements.txt
.venv-gateway/bin/python -m unittest discover -s tests -v
```

测试使用真实 localhost TCP 服务。不能仅用 httpx `ASGITransport` 验证流式行为：它会缓冲响应，不能充分验证 SSE 首块到达与真实断连。

## 2. 用模拟上游验证 4b

下面的 `dev-key-for-learning` **仅为本机教学用的公开示例，不可当作真实服务密钥**。
所有终端都先 `cd /Users/dld/locnvda`（其他机器进入各自的仓库目录）。

### 终端 A：模拟上游（已运行就不用重启）

```bash
.venv-gateway/bin/python -m uvicorn gateway.mock_upstream:app --host 127.0.0.1 --port 8000
```

模拟服务没有鉴权；保持只监听本机。它不执行模型推理，usage 数字是示意，不能用于性能/效果压测。

### 终端 B：网关

如果旧网关仍在运行，先按 Ctrl-C 停止。然后逐条运行：

```bash
export GATEWAY_API_KEYS='{"local-dev":"dev-key-for-learning"}'
.venv-gateway/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 --timeout-graceful-shutdown 10
```

`GATEWAY_API_KEYS` 是 JSON 对象，**调用方 ID → 客户端 key**。可配置多个调用方：

```bash
export GATEWAY_API_KEYS='{"app-a":"example-key-a","app-b":"example-key-b"}'
```

这行仅演示格式；若改用它，需要重启网关，并将下面 curl 中的 key 改成对应值。

没有配置 key、配置为空、格式错误、调用方 ID 重复或不同调用方使用同一 key，网关都会**拒绝启动**，不存在默认匿名放行。配置只在启动时加载；新增、撤销或轮换 key 后重启网关。

### 终端 C：客户端

无 key，预期 401：

```bash
curl -i http://127.0.0.1:8080/v1/models
```

错误 key，预期 401：

```bash
curl -i http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer wrong-key'
```

正确 key，预期 200 和模型列表：

```bash
curl -i http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer dev-key-for-learning'
```

普通 JSON（整行复制，不要在 JSON 字符串内部手动换行）：

```bash
curl -sS http://127.0.0.1:8080/v1/chat/completions -H 'Authorization: Bearer dev-key-for-learning' -H 'Content-Type: application/json' -d '{"model":"mock-model","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

SSE（`-N` 禁用 curl 输出缓冲，预期逐段文本、usage、最后 `[DONE]`）：

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Authorization: Bearer dev-key-for-learning' -H 'Content-Type: application/json' -d '{"model":"mock-model","messages":[{"role":"user","content":"你好"}],"stream":true,"stream_options":{"include_usage":true}}'
```

匿名健康检查，预期仍为 200：

```bash
curl -i http://127.0.0.1:8080/health
```

停止终端 A 后，无 key 仍返回 401（未调用上游），正确 key 返回 502（认证通过但上游不可用）。重启 A 后正常恢复，无需重启网关。

## 3. 4b 的原理

1. 在读取请求体、占用上游连接之前，先检查唯一的 `Authorization: Bearer <key>`。缺失、错误、格式不合法、重复 Authorization 都返回 401，带 `WWW-Authenticate: Bearer`，不访问上游。
2. 将匹配的 key 映射为调用方 ID，例如 `local-dev`。仅将 ID 存入 `request.state.caller_id`，为后续按调用方限流和日志做准备；忽略并删除伪造的 `X-Gateway-Caller-ID`。
3. 用固定长度的 SHA-256 摘要与 `hmac.compare_digest` 比较凭证，不直接比较原始字符串。这是高熵随机 API key，不是用户密码，不以此替代密码哈希方案。
4. **删除客户端的 Authorization**。若配置了 `GATEWAY_UPSTREAM_API_KEY`，发送 `Authorization: Bearer <上游 key>`；未配置则不发送 Authorization。客户端 key 永远不自动变成上游 key。
5. 错误响应不包含 key，配置对象 repr 隐藏凭证，不记录原始 Authorization。后续日志用调用方 ID，不用 key。key 不接受通过 query、请求体或 Cookie 传入。

**两种 key 为什么分开？**

- 客户端 key：控制哪个业务/用户能访问网关，各调用方可独立撤销。
- 上游 key：网关访问 vLLM 的凭证，不提供给客户端。
- 将来更换上游服务或上游 key，客户端无需更换自己的 key。

静态注册表适合当前学习阶段。真实 key 应由 `secrets.token_urlsafe(32)` 等生成，保存在秘密管理系统或不提交的本地配置中；不要将真实 key 写入 Git、示例、日志、URL。上游必须限制网络可达性，避免绕过网关。公网化还需要 TLS、请求大小/频率/并发限制等；API key 本身不会加密 HTTP 流量。

## 4. 接真实 vLLM

用原有 `scripts/deploy.sh <config>` 启动 vLLM，确认监听 `127.0.0.1:8000`，不要同时启动模拟服务。
请求的 `model` 用上游 `/v1/models` 返回的 ID。网关不会修改模型名、token ID、`stream_options`、`chat_template_kwargs` 等字段。

vLLM 启动脚本中的 `API_KEY` 是**上游 key**。如果启用了它，在终端 B 给网关设置同一个值（客户端 key 应使用另一个独立值）：

```bash
export GATEWAY_UPSTREAM_API_KEY='<实际的 vLLM key>'
```

vLLM 没有配置 API key 则省略这个变量，或用 `unset GATEWAY_UPSTREAM_API_KEY` 清除它。网关使用专门的变量，不会读取推理脚本的 `API_KEY`。

实例上可复用压测程序（在已安装压测依赖的 vLLM 环境中运行），除地址和凭证外保持参数一致：

```bash
# 直连使用上游凭证；vLLM 未配置 key 时不用设置 API_KEY
API_KEY='<上游 key>' python bench/bench.py --base-url http://127.0.0.1:8000 --concurrency 1,4 --input-len 1024 --output-len 256
# 经过网关使用客户端凭证
API_KEY='<客户端 key>' python bench/bench.py --base-url http://127.0.0.1:8080 --concurrency 1,4 --input-len 1024 --output-len 256
```

这一步要在真实 GPU 上进行，目前只完成本地功能验收，尚未测量网关的真实推理性能开销。

## 配置

参数通过环境变量传入，监听地址/端口由 uvicorn 的 `--host`、`--port` 控制。

| 环境变量 | 默认值 | 作用 |
|---|---|---|
| `GATEWAY_API_KEYS` | **必填** | JSON：调用方 ID → 客户端 key；非空且凭证唯一 |
| `GATEWAY_UPSTREAM_API_KEY` | 不发送上游 Authorization | 网关调用上游的独立凭证 |
| `GATEWAY_UPSTREAM_URL` | `http://127.0.0.1:8000` | 固定可信上游，不接受调用方提供的任意目标 URL |
| `GATEWAY_CONNECT_TIMEOUT` | `5` | 建立上游连接的最长等待秒数 |
| `GATEWAY_READ_TIMEOUT` | `120` | 等待响应头或下一批数据的最长空闲秒数，**不是整个请求总时限** |
| `GATEWAY_WRITE_TIMEOUT` | `30` | 写入上游数据的等待时限 |
| `GATEWAY_POOL_TIMEOUT` | `5` | 等待可用 HTTP 连接的时限 |
| `GATEWAY_MAX_CONNECTIONS` | `100` | 每个进程的上游连接池上限，**不是推理并发准入控制** |
| `GATEWAY_MAX_KEEPALIVE_CONNECTIONS` | `20` | 可保留的空闲长连接数量，不能超过连接池上限 |

调用方 ID 为 1–64 位 ASCII 字母/数字/点/下划线/连字符，首位必须是字母或数字；key 为 1–512 位 ASCII bearer token（非空、无空格或换行）。真实服务应使用足够长的随机 key，不要用短示例值。

## 4a 保留的代理行为

- 请求体完整读取，字节不改写，输入 JSON/校验归上游负责。目前没有请求体大小限制，后续公网化前必须补充。
- 普通 JSON、SSE、上游错误响应均逐块转发，不提前聚合。`aiter_raw()` 不解压，可保持 Content-Encoding / Content-Length 与实际字节一致。
- TCP/HTTP 可合并或拆分网络块；保证字节顺序与内容，而非网络 chunk 边界。SSE 的 `data:`、usage 和 `[DONE]` 不被改写。
- 过滤 Connection、Transfer-Encoding 等 hop-by-hop 头；Host 和请求 Content-Length 重新生成。除凭证/保留身份头外，端到端头（含重复 Set-Cookie）保留，不在调用方之间共享 cookies。
- 从等待上游响应头到流式结束始终监控客户端断连。结束/取消后关闭上游响应，服务退出时关闭连接池。
- 上游自身的 4xx/5xx 原样返回；发出响应头前上游请求超时返回 504，其他连接/协议错误返回 502。不跟随重定向，不自动重试。
- 流中出错时 HTTP 状态已发出，不能改成 504。中断连接并记录错误类型，不补 `[DONE]`，客户端应按失败处理。

下一步 4c：按调用方 ID 限制请求频率，并限制全局在途推理请求；4d 再补结构化日志和压测。
