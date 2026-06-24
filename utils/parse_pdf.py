import os
import json
import re
import fitz
import sys
from tqdm import tqdm
sys.path.append('..')

def clean_text(text):
    """Очистка текста от лишних переносов и мусора"""
    # Склеиваем слова, разделенные переносом на новую строку
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    
    # Убираем лишние пробелы
    text = re.sub(r'[ \t]+', ' ', text)
    
    # Сохраняем структуру абзацев (2 переноса между абзацами)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()

def chunk_text(text, chunk_size=800, overlap=150):
    """Умное разбиение текста на чанки с сохранением целостности предложений"""
    # Разбиваем на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_len = len(sentence)
        
        # Если предложение слишком длинное - разбиваем принудительно
        if sentence_len > chunk_size:
            if current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = []
                current_length = 0
            
            parts = [sentence[i:i+chunk_size] for i in range(0, len(sentence), chunk_size - overlap)]
            chunks.extend(parts)
            continue
        
        # Если добавление превысит лимит - сохраняем чанк
        if current_length + sentence_len + 1 > chunk_size and current_chunk:
            chunks.append(' '.join(current_chunk))
            
            # Оставляем overlap
            overlap_text = ' '.join(current_chunk[-max(1, overlap//50):])
            current_chunk = [overlap_text] if overlap_text else []
            current_length = len(overlap_text)
        
        current_chunk.append(sentence)
        current_length += sentence_len + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def parse_all_pdfs(data_dir="../data/", output_json_path="../data/knowledge_base.json", chunk_size=800, overlap=150):
    """Парсинг PDF с сохранением в JSON с метаданными"""
    all_chunks = []
    
    if not os.path.exists(data_dir):
        print(f"❌ Папка '{data_dir}' не найдена!")
        return
    
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ В папке '{data_dir}' не найдено PDF-файлов!")
        return

    print(f"📚 Найдено файлов для обработки: {len(pdf_files)}")

    for pdf_file in tqdm(pdf_files, desc='📄 Обрабатываем файлы'):
        pdf_path = os.path.join(data_dir, pdf_file)
        book_name = pdf_file.replace(".pdf", "").replace("_", " ").title()
        
        try:
            doc = fitz.open(pdf_path)
            print(f"\n📖 Обработка: {pdf_file} ({len(doc)} страниц)...")
            
            for page_num in tqdm(range(len(doc)), desc=f'  Страницы', leave=False):
                page = doc[page_num]
                
                # Получаем текст
                page_text = page.get_text("text")
                cleaned_text = clean_text(page_text)
                
                # Получаем таблицы (если есть)
                table_text = extract_tables_from_page(page)
                
                # Объединяем текст и таблицы
                full_text = cleaned_text
                if table_text:
                    full_text += "\n\n" + table_text
                
                if not full_text:
                    continue
                
                # Разбиваем на чанки
                chunks = chunk_text(full_text, chunk_size, overlap)
                
                for chunk in chunks:
                    if len(chunk.strip()) < 50:  # Пропускаем слишком короткие чанки
                        continue
                    
                    all_chunks.append({
                        "text": chunk.strip(),
                        "source": book_name,
                        "page": page_num + 1,
                        "file": pdf_file,
                        "chunk_length": len(chunk)
                    })
            
            doc.close()
            
        except Exception as e:
            print(f"❌ Ошибка при обработке {pdf_file}: {e}")
    
    # Сохраняем в JSON
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Успешно обработано! Создано {len(all_chunks)} чанков.")
    print(f"📁 Результат сохранён в {output_json_path}")
    
    # Статистика по источникам
    sources = {}
    for chunk in all_chunks:
        source = chunk['source']
        sources[source] = sources.get(source, 0) + 1
    
    print("\n📊 Статистика по источникам:")
    for source, count in sources.items():
        print(f"  • {source}: {count} чанков")
    
    return all_chunks

def extract_tables_from_page(page):
    """Извлекает таблицы со страницы и преобразует в текст"""
    try:
        tables = page.find_tables()
        table_texts = []
        
        for table in tables:
            df = table.to_pandas()
            # Преобразуем в читаемый текст
            text = df.to_string(index=False)
            table_texts.append(text)
        
        return "\n\n".join(table_texts) if table_texts else ""
    except:
        return ""

if __name__ == "__main__":
    # Определяем пути относительно скрипта
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # поднимаемся из utils/ в корень
    
    data_dir = os.path.join(project_root, "data")
    output_path = os.path.join(project_root, "data", "knowledge_base.json")
    
    # Создаём папку data, если её нет
    os.makedirs(data_dir, exist_ok=True)
    
    print(f"📁 Корень проекта: {project_root}")
    print(f"📁 Папка с PDF: {data_dir}")
    print(f"📄 Результат: {output_path}")
    print("-" * 50)
    
    chunks = parse_all_pdfs(
        data_dir=data_dir,
        output_json_path=output_path,
        chunk_size=800,
        overlap=150
    )
    
    if chunks:
        print(f"\n🎉 Готово! Создано {len(chunks)} чанков.")
    else:
        print("\n❌ Не удалось обработать файлы.")
    
    print("\n🎉 Готово! Теперь можно использовать knowledge_base.json для RAG.")