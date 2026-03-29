import os
from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition


# adım 1: tool'ları hazırlama
from tools_script import tools

# adım 2: modeli yükleme
llm = ChatOllama(
    model="llama3.1",
    temperature=0.1,  # 0 çok robotik olabilir, 0.1 tutarlılık ile zeka arasında iyi bir dengedir.
    num_ctx=4096,  # Bağlam penceresini geniş tutarak geçmişi unutmasını engelle.
    num_gpu=35,  # Senin sistemin için ideal GPU offloading.
    repeat_penalty=1.2,  # Modelin aynı kelimeleri tekrar etmesini (loop) engeller.
)

# modelin tool'ları bilmesini sağlamasa
llm_w_tools = llm.bind_tools(tools)


# graph durumu satate tanımlama
class AgentState(TypedDict):
    "Ajanın hafızası: mesajların listesi"

    messages: Annotated[list[BaseMessage], lambda x, y: x + y]


def call_model(state: AgentState):
    # Modelin ne yapması gerektiğini çok net söyleyen bir sistem mesajı
    system_prompt = SystemMessage(
        content=(
            "You are a Senior AI Infrastructure Engineer. Your goal is to monitor and analyze system health. "
            "Follow these strict rules:\n"
            "1. ALWAYS explain the technical meaning of the tool outputs. Do not just repeat numbers.\n"
            "2. If RAM or CPU is high, suggest specific causes (e.g., memory leaks, heavy batch size).\n"
            "3. Mandatory Language: You MUST provide your final response in TURKISH (Türkçe cevap ver).\n"
            "4. Use a professional, analytical, and helpful tone.\n"
            "5. STRUCTURE your response with these headers in Turkish: 🖥️ SİSTEM DURUMU, 🔍 ANALİZ, 💡 ÖNERİ.\n"
            "6. If you see an error or anomaly, search for it online using your search tool before providing a final answer."
        )
    )

    messages = [system_prompt] + state["messages"]
    response = llm_w_tools.invoke(messages)
    return {"messages": [response]}


# Graph oluştur and yapılandır (EDGES & LOGIC)
workflow = StateGraph(AgentState)

# workflow.add_node("agent", agent_node)
workflow.add_node("agent", call_model)
# Node for tools: Prebuilt ToolNode kullanıyoruz, araçları çalıştırır
workflow.add_node("tools", ToolNode(tools))

# Başlangıç noktası belirleme
workflow.set_entry_point("agent")

# Conditional Edges - koşullu kenar
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
    # query = """
    #         Şu an sistemin genel durumu nasıl?
    #         Önce donanım kaynaklarını kontrol et, sonra şu log satırını analiz et:
    #         'RuntimeError: CUDA out of memory in Computer Vision pipeline'
    #         """
    query = (
        "Sen uzman bir Sistem ve AI Altyapı Analistisin. "
        "Araçlardan gelen verileri ham halde bırakma, mutlaka YORUMLA. "
        "Raporunu şu başlıklarla TÜRKÇE olarak sun:\n"
        "1. 🖥️ DONANIM DURUMU: CPU, RAM ve VRAM değerlerini tek tek yaz ve limitlere yakınlığını belirt.\n"
        "2. 🔍 LOG ANALİZİ: Tespit edilen hataların ne anlama geldiğini teknik olarak açıkla.\n"
        "3. 💡 ÇÖZÜM ÖNERİSİ: Sorunu çözmek için atılması gereken somut adımları (örn: model küçültme, cache temizleme) söyle.\n"
        "Yanıtın teknik, detaylı ve profesyonel olsun."
    )

    inputs = {"messages": [HumanMessage(content=query)]}

    for event in app.stream(inputs):
        for key, value in event.items():
            print(f"\n# Adım: {key}")

            if "messages" in value:
                last_msg = value["messages"][-1]

                # Gürültüyü azaltmak için sadece içeriği veya tool_call'u yazdıralım
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    print(f"Model is calling tool: {last_msg.tool_calls}")
                else:
                    print(f"Model Result: {last_msg.content}")

    print("# # # # # #      DONE      # # # # # #")
