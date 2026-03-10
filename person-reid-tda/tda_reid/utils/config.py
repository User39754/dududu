"""
配置管理工具
"""

import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigManager:
    """
    配置管理器
    """
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get(self, key: str, default: Any = None) -> Any:
        """获取配置值"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """设置配置值"""
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, save_path: str = None):
        """保存配置"""
        path = Path(save_path or self.config_path)
        with open(path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
    
    def print_summary(self):
        """打印配置摘要"""
        print("Configuration Summary:")
        print("=" * 50)
        for key, value in self.config.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        print("=" * 50)


if __name__ == '__main__':
    cfg = ConfigManager('configs/config_market.yaml')
    cfg.print_summary()
