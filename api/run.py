import sys
import os

# 添加当前目录到 Python 路径，以便导入模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn

if __name__ == "__main__":
    print("Starting FastAPI server...")
    uvicorn.run("api.index:app", host="127.0.0.1", port=8001, reload=True)
