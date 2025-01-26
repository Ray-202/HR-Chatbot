
from django.shortcuts import render
from django.http import JsonResponse
from langchain.llms import Ollama
from langchain_community.vectorstores import Pinecone as PineconeStore
from .rag import embeddings, Index_Name
import time
from django.core.cache import cache

def chatbot_interface(request):
    return render(request, 'chatbot.html')

def chatbot_view(request):
   
    user_query = request.GET.get('query', '')
    if not user_query:
        return JsonResponse({"error": "No query provided."}, status=400)
     # Check cache
    cache_key = f"response_{user_query}"
    cached_response = cache.get(cache_key)
    if cached_response:
        return JsonResponse({"answer": cached_response})

    start_time = time.time()

    # 1. Setup Ollama LLM 
    ollama_llm = Ollama(model="prakasharyan/qwen-arabic")

    # 2. Recreate Pinecone vectorstore to do similarity search
    vectorstore = PineconeStore.from_existing_index(
    index_name=Index_Name,
    embedding=embeddings
    )

  
    similar_docs = vectorstore.similarity_search(user_query, k=2)
    print("Retrieved Chunks:", similar_docs)
    # 4. Build a prompt context from those docs
    context_text = "\n".join([doc.page_content[:500] for doc in similar_docs])
    prompt = f"""
أنت خبير في الموارد البشرية. يمكنك الإجابة فقط على الأسئلة المتعلقة بالموارد البشرية.
إذا كان السؤال غير متعلق بالموارد البشرية، أجب بما يلي:
"عذرًا، أنا روبوت خاص بالموارد البشرية ولا يمكنني الإجابة على أسئلة غير متعلقة بالموارد البشرية."
وإذا لم يتضمن السياق معلومات كافية للإجابة، قل:
"عذرًا، لا توجد معلومات كافية للإجابة على هذا السؤال."

السياق:
{context_text}

السؤال: {user_query}

أجب فقط باستخدام المعلومات الموجودة في السياق، ولا تضف أي معلومات خارجية أو تخمينات:
"""






    # Get LLM response
    llm_response = ollama_llm.invoke(prompt)


    # Check if response is a string
    if isinstance(llm_response, str):
        answer = llm_response  # Directly use the string response
    elif isinstance(llm_response, dict):
        answer = llm_response.get("text", "No response from LLM.")
    else:
        answer = "Unexpected response type from LLM."
    return JsonResponse({"answer": answer})
