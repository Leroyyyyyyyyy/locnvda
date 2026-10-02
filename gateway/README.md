# 阶段 4a / 4b / 4c：代理、鉴权、限流与并发保护

已实现 JSON / SSE 代理、调用方 API key 鉴权、上游凭证隔离、按调用方令牌桶限流和全局推理请求准入。**限流/并发状态仅在单个进程内有效，没有请求体大小限制，继续只绑定本机，不要直接暴露到公网。**

```text
客户端（客户端 key） → 网关 :8080（认证 → 限流 → 并发准入） → vLLM :8000（上游 key）
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
.venv-gateway/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 --workers 1 --timeout-graceful-shutdown 10
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

静态注册表适合当前学习阶段。真实 key 应由 `secrets.token_urlsafe(32)` 等生成，保存在秘密管理系统或不提交的本地配置中；不要将真实 key 写入 Git、示例、日志、URL。上游必须限制网络可达性，避免绕过网关。公网化还需要 TLS、请求大小限制、可跨进程的资源保护等；现有单进程内存限流不代表已具备生产安全能力，API key 本身也不会加密 HTTP 流量。

## 4. 4c：限流与并发保护

### 两种限制不同

- **频率限制**：每个调用方一个令牌桶。桶启动时装满 `burst` 个令牌，通过一次频率检查扣 1 个，每秒补充 `rps` 个，最多存到 `burst`。默认平均 5 req/s、允许最多 10 个积攒的突发请求；不是固定窗口内绝不超过 5 个。
- **并发限制**：所有调用方共用 `max_in_flight` 个推理名额，默认 8 个。名额从读取请求体前一直占到响应与上游清理结束。SSE 发出第一块后仍占着，不能提前释放。

| 路由 | 需要鉴权 | 消耗调用方频率额度 | 占推理名额 |
|---|---|---|---|
| `/v1/models` | 是 | 是 | 否 |
| 两个 POST 生成接口 | 是 | 是 | 是 |
| `/health` | 否 | 否 | 否 |

处理顺序：鉴权 → 频率检查/扣令牌 → 推理名额检查 → 读请求体 → 调用上游。

- 频率超限：**429**，`error.code=rate_limit_exceeded`，`Retry-After` 为补足 1 个令牌需要的秒数，向上取整。
- 推理名额已满：**503**，`error.code=gateway_busy`，立即拒绝不排队；`Retry-After: 1` 是重试提示，不保证 1 秒后一定有容量。
- 通过频率检查即扣令牌，包括随后收到 503、上游报错/超时或客户端断连的请求，不退还。429 拒绝不扣令牌；无效 key 不消耗任何调用方额度。
- 推理名额会在完成、错误、超时或断连时释放；重复释放不会增加容量。上传中途断连同样释放，返回 499 是内部约定，已断开的客户端不会收到它。

### 4.1 手动观察 429

终端 A 保持模拟服务运行。终端 B 按 Ctrl-C 停止网关，再逐条执行：

```bash
export GATEWAY_API_KEYS='{"local-dev":"dev-key-for-learning"}'
export GATEWAY_RATE_LIMIT_RPS=0.1
export GATEWAY_RATE_LIMIT_BURST=2
export GATEWAY_MAX_IN_FLIGHT=1
.venv-gateway/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 --workers 1 --timeout-graceful-shutdown 10
```

这相当于桶里先有 2 张通行票，每 10 秒补 1 张。
终端 C 复制整个循环，连续发 3 次请求：

```bash
for i in 1 2 3; do curl -sS -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer dev-key-for-learning'; done
```

没有其他调用时预期为 `200 / 200 / 429`。等待约 10 秒后再执行一次：

```bash
curl -i http://127.0.0.1:8080/v1/models -H 'Authorization: Bearer dev-key-for-learning'
```

预期恢复 200；再次紧接着请求则可能是 429，可看到 `Retry-After`。
如需观察不同调用方互不影响，在 `GATEWAY_API_KEYS` 中增加另一个调用方/key 并重启；一个调用方耗尽额度不会耗尽另一个的额度。

### 4.2 手动观察 503 和名额释放

为避免 429 干扰，先放宽频率额度，但把并发名额保持为 1。

终端 A 按 Ctrl-C 停止旧模拟服务，重新启动更慢的模拟流（3 段，约 9 秒结束）：

```bash
MOCK_STREAM_DELAY=3 .venv-gateway/bin/python -m uvicorn gateway.mock_upstream:app --host 127.0.0.1 --port 8000
```

`MOCK_STREAM_DELAY` 仅为演示参数，默认每段等待 0.3 秒；不影响真实 vLLM。

终端 B 停止网关后改配置并重启（保留刚才的 `GATEWAY_API_KEYS`）：

```bash
export GATEWAY_RATE_LIMIT_RPS=100
export GATEWAY_RATE_LIMIT_BURST=100
export GATEWAY_MAX_IN_FLIGHT=1
.venv-gateway/bin/python -m uvicorn gateway.app:app --host 127.0.0.1 --port 8080 --workers 1 --timeout-graceful-shutdown 10
```

终端 C 开始 SSE 请求：

```bash
curl -N http://127.0.0.1:8080/v1/chat/completions -H 'Authorization: Bearer dev-key-for-learning' -H 'Content-Type: application/json' -d '{"model":"mock-model","messages":[{"role":"user","content":"你好"}],"stream":true}'
```

**新开终端 D**，在 C 还未结束时发起另一个生成请求（GET 模型列表不占推理名额，不能用来验证这个 503）：

```bash
curl -i http://127.0.0.1:8080/v1/completions -H 'Authorization: Bearer dev-key-for-learning' -H 'Content-Type: application/json' -d '{"model":"mock-model","prompt":"你好","stream":false}'
```

预期返回 503 `gateway_busy`。C 正常结束后，D 重复同一条命令应恢复 200。
也可以重新开始 C 的 SSE，在它输出期间按 Ctrl-C 中断；稍后再发 D 的请求也应恢复 200，说明断连释放了名额。

### 实现边界

`gateway/limits.py` 使用单调时钟计时，仅为预先配置的调用方分配桶；检查/修改之间没有 `await`，在单个事件循环中不会因协程切换而超卖名额。**只运行一个 worker**。多 worker、多副本分别拥有各自额度与并发计数，实际总额度会放大；重启也会重置桶。此版本没有 Redis 共享状态，也不支持跨线程共享控制器。

配置的每调用方限额相同，暂不支持套餐差异、按输入/输出 token 计费或按请求成本加权。并发名额也不是 vLLM 的真实 Running 序列数：输入长度、输出长度和请求的 `n` 都会影响 GPU 成本。默认 8 仅为教学起点，不代表某个 GPU 的生产推荐值；要按目标 p99 与工作负载实测。

HTTP 连接池与推理名额是不同限制：`max_in_flight` 不能大于 `max_connections`，建议连接池留额外空间给模型列表。`/v1/models` 不因推理名额已满被直接拒绝，但仍受调用方频率、HTTP 连接池和上游状态影响。

## 5. 接真实 vLLM

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
| `GATEWAY_RATE_LIMIT_RPS` | `5` | 每个调用方每秒补充的请求令牌数，支持正小数 |
| `GATEWAY_RATE_LIMIT_BURST` | `10` | 每个调用方桶容量/启动令牌数，正整数 |
| `GATEWAY_MAX_IN_FLIGHT` | `8` | 每进程最大在途推理请求数，正整数，不超过连接池上限 |

调用方 ID 为 1–64 位 ASCII 字母/数字/点/下划线/连字符，首位必须是字母或数字；key 为 1–512 位 ASCII bearer token（非空、无空格或换行）。真实服务应使用足够长的随机 key，不要用短示例值。

## 4a 保留的代理行为

- 请求体完整读取，字节不改写，输入 JSON/校验归上游负责。目前没有请求体大小限制，后续公网化前必须补充。
- 普通 JSON、SSE、上游错误响应均逐块转发，不提前聚合。`aiter_raw()` 不解压，可保持 Content-Encoding / Content-Length 与实际字节一致。
- TCP/HTTP 可合并或拆分网络块；保证字节顺序与内容，而非网络 chunk 边界。SSE 的 `data:`、usage 和 `[DONE]` 不被改写。
- 过滤 Connection、Transfer-Encoding 等 hop-by-hop 头；Host 和请求 Content-Length 重新生成。除凭证/保留身份头外，端到端头（含重复 Set-Cookie）保留，不在调用方之间共享 cookies。
- 从等待上游响应头到流式结束始终监控客户端断连。结束/取消后关闭上游响应，服务退出时关闭连接池。
- 上游自身的 4xx/5xx 原样返回；发出响应头前上游请求超时返回 504，其他连接/协议错误返回 502。不跟随重定向，不自动重试。
- 流中出错时 HTTP 状态已发出，不能改成 504。中断连接并记录错误类型，不补 `[DONE]`，客户端应按失败处理。

下一步 4d：结构化日志与网关压测，分别记录鉴权失败、限流拒绝、并发拒绝和上游失败；真实 GPU 上再调整频率/并发参数并测量尾延迟。
