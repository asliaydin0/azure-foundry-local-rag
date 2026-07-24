from foundry_local_sdk import FoundryLocalManager, Configuration

def main():
    print("🚀 Yerel Yapay Zeka motoru başlatılıyor...")
    
    try:
        config = Configuration("azure-foundry-local-rag") 
        manager = FoundryLocalManager(config)
        
        hedef_model_id = "Phi-3.5-mini-instruct-generic-cpu:2"
        modeller = manager.catalog.list_models()
        secili_model = next((m for m in modeller if m.id == hedef_model_id), None)
        
        print("⏳ Model hafızaya alınıyor (İnternetsiz)...")

        secili_model.load()
        print("✅ Model başarıyla yüklendi!")
        
        # Sohbet istemcisini çağırıyoruz
        chat_client = secili_model.get_chat_client()
        
        soru = "Merhaba! Sen kimsin ve şu an internetsiz mi çalışıyorsun? Lütfen bana çok kısa ve Türkçe cevap ver."
        print(f"\nSiz: {soru}")
        print("🤖 Asistan düşünüyor...\n")
        
        # Bulduğumuz doğru metodu (complete_chat) kullanıyoruz
        response = chat_client.complete_chat(
            messages=[
                {"role": "system", "content": "Sen yerel bilgisayarda çalışan, dürüst ve Türkçe konuşan yardımsever bir asistansın."},
                {"role": "user", "content": soru}
            ]
        )
        
        # Gelen cevabın yapısını kontrol edip güvenli bir şekilde ekrana yazdırıyoruz
        if hasattr(response, 'choices'):
            cevap = response.choices[0].message.content
        else:
            # Beklenmedik bir obje dönerse diye ham halini yakalıyoruz
            cevap = str(response)
            
        print(f"🤖 Asistan: {cevap}")
        print("\n🎉 TEBRİKLER! Hello AI aşaması başarıyla tamamlandı.")
        
    except Exception as e:
        print(f"\n❌ Bir hata oluştu: {e}")

if __name__ == "__main__":
    main()