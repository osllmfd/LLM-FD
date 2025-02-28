# LLM-FD - Large Language Model for Fault Detection

Industrial fault detection powered by ​**GLM-4-9B**​ model and ​**Dify**​ workflow orchestration.

![Dify Integration](https://img.shields.io/badge/Dify-0.9%2B-blue)
![GLM-4](https://img.shields.io/badge/GLM--4-9B-orange)

## 📋 Core Components

### Essential References
- ​**Dify Platform**: [https://github.com/langgenius/dify](https://github.com/langgenius/dify)
- ​**GLM-4-9B-chat Model**: [https://github.com/THUDM/GLM-4](https://github.com/THUDM/GLM-4)
- ​**Fault Detection Workflow**: `fault_detection.yml` (Core configuration)
- ​**Python Environment**: Python 3.11 and `requirements.txt` (use cmd 'pip install -r requirements.txt' to install)

## 🚀 Quick Start

### 1. Import Workflow to Dify
[Import local DSL file](https://docs.dify.ai/guides/application-orchestrate/creating-an-application#import-local-dsl-file)
  
api_key: [Access API Reference](https://docs.dify.ai/guides/workflow/publish)

### 2. Start GLM Http Serve
download **[GLM-4-9B-chat](https://github.com/THUDM/GLM-4)** model file to ./models/glm/

python run_glm_http_serve.py

### 3. Fault Detection
use your api_key in run_fd.py
'python run_fd.py' for fault detection
