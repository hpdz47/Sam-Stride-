from autogen.tools.experimental import DeepResearchTool
#from vLLM_Manager import LLM_Manager
from vLLM_Configuration import VLLM_Config
import sys
import logging
import os
from dotenv import load_dotenv
import socket
import subprocess


if __name__ == "__main__":
    load_dotenv('/research_scripts/.env')

    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    print(f"Hostname: {hostname} | IP Address: {ip_address}")



    os.environ['PATH'] = '/pip/bin:' + os.environ.get('PATH', '/usr/local/bin:/usr/bin:/bin')
    print("Starting Deep Research")
    
    
    Deep_Research_llm_config=VLLM_Config(api_type="openai",cache_seed=None,temperature=0.4,enable_thinking=False,LLM_Type="VL",IP_Address=ip_address).build_config()

    Deep_Research_Tool=DeepResearchTool(llm_config=Deep_Research_llm_config,max_web_steps=25)
    result=Deep_Research_Tool.func(task=sys.argv[1])

    # Save result:
    output_path = "/research/Deep_Research_Report.md"
    with open(output_path, "a") as f:
        #f.write(f"\n\n# Deep Research: {task}\n\n")
        
        # Extract detailed research from the chat history
        chat_history = Deep_Research_Tool.critic_agent.chat_messages
        for agent_name, messages in chat_history.items():
            for msg in messages:
                content = msg.get('content', '')
                if content and isinstance(content, str):
                    # Include tool responses that contain "Subquestions answered:"
                    if "Subquestions answered:" in content:
                        f.write("## Detailed Research\n\n")
                        f.write(content)
                        f.write("\n\n---\n\n")
        
        # Add the final summary
        f.write("## Final Summary\n\n")
        f.write(result)
        f.write("\n")
    
    print(f"Saved results to {output_path}")