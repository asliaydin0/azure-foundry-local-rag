from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    print("🚀 Yerel Yapay Zeka motoru başlatılıyor...")
    
    try:
        config = Configuration("azure-foundry-local-rag") 
        manager = FoundryLocalManager(config)
        
        # 1. Sohbet için kullanacağımız modeli seçiyoruz
        hedef_model_id = "Phi-3.5-mini-instruct-generic-cpu:2"
        print(f"🧠 {hedef_model_id} katalogda aranıyor...")
        
        # Modeli ID'sine göre listeden buluyoruz
        modeller = manager.catalog.list_models()
        secili_model = next((m for m in modeller if m.id == hedef_model_id), None)
        
        if not secili_model:
            print("❌ Model katalogda bulunamadı!")
            return

        # 2. Modeli indir ve RAM'e yükle
        print("\n⏳ Model hazırlanıyor (İndirme/Yükleme)...")
        print("💡 Not: Model bilgisayara ilk kez iniyorsa, internet hızına bağlı olarak bu işlem birkaç dakika sürebilir. Lütfen terminali kapatma.")
        
        secili_model.download() # Sadece ilk seferde indirir, sonrakilerde önbellekten (cache) okur
        secili_model.load()     # Modeli hafızaya alır
        print("✅ Model başarıyla yüklendi ve internetsiz çalışmaya hazır!")
        
        # 3. Sohbet istemcisini oluşturuyoruz (OpenAI standartlarında çalışır)
        chat_client = secili_model.get_chat_client()
        
        # 4. İlk sorumuzu soruyoruz
        soru = "Merhaba! Sen kimsin ve şu an internetsiz mi çalışıyorsun? Lütfen bana çok kısa ve Türkçe cevap ver."
        print(f"\nSiz: {soru}")
        
        # Modelden cevabı alıyoruz
        response = chat_client.chat.completions.create(
            model=hedef_model_id,
            messages=[
                {"role": "system", "content": "Sen yerel bilgisayarda çalışan, dürüst ve Türkçe konuşan yardımsever bir asistansın."},
                {"role": "user", "content": soru}
            ]
        )
        
        # Gelen cevabı ekrana yazdırıyoruz
        cevap = response.choices[0].message.content
        print(f"\n🤖 Asistan: {cevap}")
        
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()