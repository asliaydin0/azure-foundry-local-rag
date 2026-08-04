from retriever import VectorRetriever

def run_rag_pipeline():
    print("🚀 RAG (Retrieval-Augmented Generation) Sistemi Başlatılıyor...\n")
    
    # 1. Arama Motorunu Yükle (FoundryLocalManager burada arka planda başlatılır)
    retriever = VectorRetriever()
    
    # Singleton kuralı gereği retriever içindeki manager'ı kullanıyoruz
    manager = retriever.embedder.manager
    
    hedef_model_id = "Phi-3.5-mini-instruct-generic-cpu:2"
    modeller = manager.catalog.list_models()
    llm_model = next((m for m in modeller if m.id == hedef_model_id), None)
    
    print(f"\n🧠 Dil Modeli ({hedef_model_id}) hafızaya alınıyor...")
    llm_model.load()
    chat_client = llm_model.get_chat_client()
    print("✅ Dil Modeli hazır!\n")
    
    # 2. Kullanıcı Sorusu
    soru = "Mobil uygulama geliştirirken arayüz donmalarını nasıl engelleyebilirim?"
    print(f"👤 Kullanıcı: {soru}\n")
    
    # 3. Veritabanında Arama (Retrieval)
    print("🔍 Veritabanından bağlam (context) aranıyor...")
    bulunan_dokumanlar = retriever.search(soru, top_k=2)
    
    baglam_metni = "\n\n".join([f"- Kaynak: {doc['source']}\n  İçerik: {doc['text']}" for doc in bulunan_dokumanlar])
    print("✅ İlgili bilgiler veritabanından çekildi.\n")
    
    # 4. Prompt Mühendisliği
    system_prompt = f"""Sen uzman bir yazılım asistanısın. 
Kullanıcının sorusunu SADECE aşağıdaki BAĞLAM bölümünde verilen bilgileri kullanarak cevapla.
Eğer bağlamda sorunun cevabı yoksa, 'Bu konuda bilgim yok' de ve kesinlikle kendi bilgini uydurma (halüsinasyon yapma).
Cevabını verirken mutlaka hangi dosyadan (kaynaktan) yararlandığını belirt.

BAĞLAM:
{baglam_metni}
"""
    
    # 5. LLM Ayarları ve Cevap Üretimi
    print("🤖 Asistan dokümanları okuyor ve cevabı üretiyor...\n")
    
    # RAM'in patlamasını engellemek için ayarları önden tanımlıyoruz
    chat_client.settings.max_tokens = 300
    chat_client.settings.temperature = 0.1
    
    try:
        response = chat_client.complete_chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": soru}
            ]
        )
        
        # Gelen cevabı al ve ekrana yazdır
        if hasattr(response, 'choices'):
            cevap = response.choices[0].message.content
        else:
            cevap = str(response)
            
        print("="*50)
        print(f"🤖 ASİSTANIN CEVABI:\n\n{cevap}")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Cevap üretilirken hata oluştu: {e}")

if __name__ == "__main__":
    run_rag_pipeline()