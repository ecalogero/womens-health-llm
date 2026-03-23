from dotenv import load_dotenv
load_dotenv()
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from llama_index.llms.openai import OpenAI
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.tools.tavily_research import TavilyToolSpec
import os
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
    AgentStream,
)

##############################################################

llm_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-4-mini-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
#tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-4-mini-instruct")

#pipe = pipeline(
#    "text-generation",
#    model=llm_model,
#    tokenizer=tokenizer,
#    batch_size = 16,
#)

#generation_args = {
#    "max_new_tokens": 500,
#    "return_full_text": False,
#    "do_sample": False,
#}

#prompt = [
#        {"role": "system", "content": " "},
#        {"role": "user", "content": f""},
#        ]
#output = pipe(prompt, **generation_args)

#topic_list_string = output[0]['generated_text']

##########################################################################

tavily_tool = TavilyToolSpec( api_key=os.getenv("TAVILY_API_KEY") )

workflow = AgentWorkflow.from_tools_or_functions(
    tavily_tool.to_tool_list(),
    llm=llm_model,
    system_prompt="You're an expert on women's reproductive health who is able to explain medical terms in simple terms to speakers of English as a second language."
)

async def main():
    handler = workflow.run(user_msg="I feel bloated and have cramps. What could be causing this?", stream=True)

    # handle streaming output
    async for event in handler.stream_events():
        if isinstance(event, AgentStream):
            print(event.delta, end="", flush=True)
        elif isinstance(event, AgentInput):
            print("Agent input: ", event.input)  # the current input messages
            print("Agent name:", event.current_agent_name)  # the current agent name
        elif isinstance(event, AgentOutput):
            print("Agent output: ", event.response)  # the current full response
            print("Tool calls made: ", event.tool_calls)  # the selected tool calls, if any
            print("Raw LLM response: ", event.raw)  # the raw llm api response
        elif isinstance(event, ToolCallResult):
            print("Tool called: ", event.tool_name)  # the tool name
            print("Arguments to the tool: ", event.tool_kwargs)  # the tool kwargs
            print("Tool output: ", event.tool_output)  # the tool output            

    # print final output
    print(str(await handler))

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
