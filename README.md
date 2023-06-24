# CRSLab

This branch integrates the new evaluation approach from iEvaLM, currently supporting ChatGPT models and ReDial dataset. We will complete the evaluation for all models and datasets in the future. Please continue to follow us!

# Quick Start 🚀

## Modify your OpenAI API key 🔑
Please open the config/iEvaLM/chatgpt/redial.yaml file and replace the **your_api_key** with your own OpenAI API key.

## Evaluate 🤔
You can use two types of interaction: attribute-based question answering and free-form chit-chat.
```bash
bash chatgpt_chat.sh # free-form chit-chat
bash chatgpt_ask.sh # attribute-based question answering
```