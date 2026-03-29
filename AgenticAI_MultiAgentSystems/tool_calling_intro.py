import os
from typing import Annotated, Literal, TypedDict

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

# Memory Saver sınıfı LangGraph'ın bellek kullanımını aktif etmek içindir,
#  model geçmiş konuşmalrı hatılayarak değerlendirmelerde bulunabilir.
from langgraph.checkpoint.memory import MemorySaver

from hardware_guard import HardwareGuard

# adım 1: tool'ları hazırlama
from analyst_tools import tools

# Bellek tabanlı hafıza
memory = MemorySaver()

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
# app = workflow.compile()
# Bellekli yapı eklenmesi için güncellendi
app = workflow.compile(checkpointer=memory)


def run_interactive_agent():
    # Her konuşma bir 'thread_id' ile takip edilir. Bu sayede ajan konuşan kişiyi hatırlar.
    config = {"configurable": {"thread_id": "sistem_analisti_01"}}

    # print("\n🤖 [Sistem Analisti Ajanı Başlatıldı]")
    print("\n🚀 [LSA-Agent] Aktif. Terminalden konuşabilirsiniz.")
    print("Çıkmak için 'exit' veya 'quit' yazabilirsin.\n")

    while True:
        user_input = input("\nSiz: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        # Kullanıcı mesajını gönder ve yanıtı işle
        inputs = {"messages": [HumanMessage(content=user_input)]}

        # config buraya parametre olarak verilir
        # LangGraph bu id ile SQLite/Memory içindeki eski mesajları bulur.
        # stream_mode="values" kullanarak tüm konuşma akışını alalım
        for event in app.stream(inputs, config=config, stream_mode="values"):
            # Sadece modelin yazdığı en son mesajı yakala
            if "messages" in event:
                last_msg = event["messages"][-1]

                # 1. Eğer model bir ARAÇ ÇAĞRISI yapıyorsa, kullanıcıya "Analiz ediliyor..." de
                if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
                    tool_name = last_msg.tool_calls[0]["name"]
                    print(f"🛠️ [SİSTEM] {tool_name} verileri toplanıyor...")

                # 2. Eğer mesaj modelden geliyorsa (AI) ve içeriği varsa (JSON DEĞİLSE)
                # Sadece son metin cevabını basar
                elif (
                    isinstance(last_msg, AIMessage)
                    and last_msg.content
                    and not last_msg.tool_calls
                ):
                    # Model bazen hala JSON basmaya çalışırsa diye küçük bir kontrol
                    if "{" not in last_msg.content[:10]:
                        print(f"\n📢 AJAN:\n{last_msg.content}")


# Test - Execute Workflow
if __name__ == "__main__":
    # check system hardware
    HardwareGuard().check_capabilities()

    # run Local System Analyst
    run_interactive_agent()
