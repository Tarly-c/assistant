import os
import sys
import threading
import importlib.util
from typing import Dict, Any, List, Optional
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class SkillRegistry:
    """线程安全的技能注册中心"""
    def __init__(self):
        self._skills: Dict[str, Any] = {}
        # 引入重入锁，防止一边读取一边写入导致内存崩溃
        self._lock = threading.RLock()

    def update_skill(self, filename: str, skill_def: dict):
        with self._lock:
            # Pydantic 自动生成 Schema
            parameters_schema = skill_def["args_schema"].model_json_schema()
            llm_schema = {
                "type": "function",
                "function": {
                    "name": skill_def["name"],
                    "description": skill_def["description"],
                    "parameters": parameters_schema
                }
            }
            # 将 filename 作为 key 绑定，方便删除时追踪
            self._skills[filename] = {
                "name": skill_def["name"],
                "func": skill_def["func"],
                "args_schema": skill_def["args_schema"],
                "llm_schema": llm_schema
            }
            print(f"[✅ 技能挂载] 成功加载/热更新: {skill_def['name']} (来源: {filename})")

    def remove_skill(self, filename: str):
        with self._lock:
            if filename in self._skills:
                skill_name = self._skills[filename]['name']
                del self._skills[filename]
                print(f"[🗑️ 技能卸载] 成功移除技能: {skill_name}")

    def get_all_llm_schemas(self) -> List[dict]:
        """供 Agent 主循环获取当前所有可用的 Schema (读取时加锁拷贝)"""
        with self._lock:
            return [info["llm_schema"] for info in self._skills.values()]

    def get_skill_info(self, skill_name: str) -> Optional[dict]:
        with self._lock:
            for info in self._skills.values():
                if info["name"] == skill_name:
                    return info
            return None


class SkillHotReloader(FileSystemEventHandler):
    """底层文件系统事件监听器"""
    def __init__(self, registry: SkillRegistry, folder_path: str):
        self.registry = registry
        self.folder_path = folder_path

    def _load_module(self, filepath: str):
        filename = os.path.basename(filepath)
        module_name = f"dynamic_skill_{filename[:-3]}"

        try:
            # 🚨 核心防御：强制清理 Python 内部模块缓存，防止 reload 失败或内存泄漏
            if module_name in sys.modules:
                del sys.modules[module_name]

            spec = importlib.util.spec_from_file_location(module_name, filepath)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, "SKILL_DEF"):
                self.registry.update_skill(filename, module.SKILL_DEF)
            else:
                print(f"[⚠️ 警告] {filename} 缺少 SKILL_DEF 契约，跳过加载。")
        except Exception as e:
            print(f"[❌ 致命错误] 动态加载 {filename} 失败: {str(e)}")

    # 监听文件修改和新建
    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            self._load_module(event.src_path)

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            self._load_module(event.src_path)

    # 监听文件删除
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.py'):
            filename = os.path.basename(event.src_path)
            self.registry.remove_skill(filename)

def start_watchdog(registry: SkillRegistry, folder_path: str = "skills"):
    """启动后台监听守护线程"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    # 先进行一次全量初始化加载
    event_handler = SkillHotReloader(registry, folder_path)
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            event_handler._load_module(os.path.join(folder_path, filename))

    # 启动后台观测器
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.daemon = True # 设置为守护线程，主进程退出时自动销毁
    observer.start()
    print(f"👀 [系统] Watchdog 已启动，正在后台监听 '{folder_path}' 目录的热更新...")
    return observer