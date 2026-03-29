import os
from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


# adım 1: tool'ları hazırlama
from tools_script import tools

# adım 2: modeli yükleme
llm = ChatOllama(model="llama3.1", temperature=0)

# modelin tool'ları bilmesini sağlamasa
llm_w_tools = llm.bind_tools(tools)


# graph durumu satate tanımlama
class AgentState(TypedDict):
    "Ajanın hafızası: mesajların listesi"

    messages: Annotated[list[BaseMessage], lambda x, y: x + y]


# düğümlerin tanımlanması
def agent_node(state: AgentState):
    messages = state["messages"]
    response = llm_w_tools.invoke(messages)
    return {"messages": [response]}


# Araç Düğümü: Prebuilt ToolNode kullanıyoruz, araçları çalıştırır
tool_node = ToolNode(tools)

# graphı oluştur yapılandır (EDGES & LOGIC)
workflow = StateGraph(AgentState)

workflow.add_node("agent", agent_node)
workflow.add_node("tools", tool_node)

# Başlangıç noktası belirleme
workflow.set_entry_point("agent")

# Conditionla Edges - koşullu kenar
# Model cevap verdikten sonra ne
# yapacağına karar verir (araç mı çağırsın, bitirsin mi?)
workflow.add_conditional_edges(
    "agent",
    # modelin cevabında "tool_calls" var mı yok mu?
    tools_condition,
)

# Araçlar çalıştıktan sonra tekrar agent'a sonucu değerlendir
workflow.add_edge("tools", "agent")

# Graph'ı derle
app = workflow.compile()


# Test - Execute Workflow
if __name__ == "__main__":
    print("LangGraph Llama3 Tool Calling Başlıyor")
    # query = "Merhaba lütfen bana 200 ile 2 yi çarpar mısın?"
    query = """
            Şu an sistemin genel durumu nasıl? 
            Önce donanım kaynaklarını kontrol et, sonra şu log satırını analiz et: 
            'RuntimeError: CUDA out of memory in Computer Vision pipeline'
            """

    inputs = {"messages": [HumanMessage(content=query)]}

    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\n# Adım: {key}")
            if "messages" in value:
                last_msg = value["messages"][-1]
                # Gürültüyü azaltmak için sadece içeriği veya tool_call'u yazdıralım
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"Model Aaç Çağırıyor: {last_msg.tool_calls}")
                else:
                    print(f"Model cevabı: {last_msg.content}")

    print("Akış Tamamlandı")
