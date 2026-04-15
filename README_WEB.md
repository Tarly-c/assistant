# Medical Assistant Web UI

这套文件用于把当前仓库里的命令行版助手，扩展成一个可被浏览器访问的网页版本。

## 包含内容

- `web_server.py`：FastAPI 后端，调用现有 `app.py` 里的 LangGraph 流程
- `web_static/index.html`：单页前端，展示聊天、plan、本地命中、PubMed 命中
- `requirements-web.txt`：网页层额外依赖
- `deploy/nginx.mysuperlt.top.conf`：Nginx 反向代理示例
- `deploy/assistant-web.service`：systemd 服务示例

## 页面能力

- 浏览器访问聊天
- 展示清洗后的 plan
- 展示本地命中：文件名、chunk、分数、摘要
- 展示 PubMed 命中：PMID、期刊、日期、重排分数、摘要
- 页面关闭或刷新时，自动结束服务端会话

## 集成方式

把这些文件放到你的仓库根目录：

- `web_server.py`
- `web_static/index.html`
- `requirements-web.txt`
- `deploy/nginx.mysuperlt.top.conf`
- `deploy/assistant-web.service`

## 本地启动

先确保你原项目依赖已经安装完成，并且 Ollama 与向量库能正常运行。

```bash
pip install -r requirements-web.txt
uvicorn web_server:web_app --host 0.0.0.0 --port 8000 --reload
```

然后打开：

```text
http://127.0.0.1:8000
```

## 部署到 `www.mysuperlt.top`

### 1. 域名解析

把：

- `@`
- `www`

都解析到你的服务器公网 IP。

### 2. 上传项目

把整个项目放到服务器，例如：

```bash
/opt/assistant
```

### 3. 创建虚拟环境并安装依赖

```bash
cd /opt/assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-web.txt
```

### 4. 启动 systemd 服务

把 `deploy/assistant-web.service` 复制到：

```bash
/etc/systemd/system/assistant-web.service
```

按实际路径修改：

- `User`
- `WorkingDirectory`
- `ExecStart`

然后执行：

```bash
sudo systemctl daemon-reload
sudo systemctl enable assistant-web
sudo systemctl start assistant-web
sudo systemctl status assistant-web
```

### 5. 配置 Nginx

把 `deploy/nginx.mysuperlt.top.conf` 复制到：

```bash
/etc/nginx/sites-available/mysuperlt.top
```

启用：

```bash
sudo ln -s /etc/nginx/sites-available/mysuperlt.top /etc/nginx/sites-enabled/mysuperlt.top
sudo nginx -t
sudo systemctl reload nginx
```

### 6. 配置 HTTPS

建议用 Certbot：

```bash
sudo certbot --nginx -d mysuperlt.top -d www.mysuperlt.top
```

## 说明

当前网页层会维护“浏览器会话”，关闭或刷新页面时会自动通知后端结束该会话。

如果你后面要把真正的多轮上下文记忆做进 LangGraph，可以继续沿用这个 `session_id` 设计，把消息历史正式传入 graph state。
