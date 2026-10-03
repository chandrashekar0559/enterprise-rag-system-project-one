from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader, JSONLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader

def load_documents(file_path: str) -> List[Any]:
    data_path = Path(file_path).resolve()
    print(f"Loading documents from: {data_path}")
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")
    documents = []
    pdf_files = list(data_path.glob("**/*.pdf"))  
    print(f"Found {len(pdf_files)} PDF files.")
    for pdf_file in pdf_files:
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents.extend(loader.load())
            print(f"Loaded {len(documents)} documents from {pdf_file}")
        except Exception as e:
            print(f"Error loading {pdf_file}: {e}")

    txt_files = list(data_path.glob("**/*.txt"))
    print(f"Found {len(txt_files)} TXT files.")
    for txt_file in txt_files:
        try:
            loader = TextLoader(str(txt_file))
            documents.extend(loader.load())
            print(f"Loaded {len(documents)} documents from {txt_file}")
        except Exception as e:
            print(f"Error loading {txt_file}: {e}")

    csv_files = list(data_path.glob("**/*.csv"))
    print(f"Found {len(csv_files)} CSV files.")
    for csv_file in csv_files:
        try:
            loader = CSVLoader(str(csv_file))
            documents.extend(loader.load())
            print(f"Loaded {len(documents)} documents from {csv_file}")
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")

    docx_files = list(data_path.glob("**/*.docx"))
    print(f"Found {len(docx_files)} DOCX files.")
    for docx_file in docx_files:
        try:
            loader = Docx2txtLoader(str(docx_file))
            documents.extend(loader.load())
            print(f"Loaded {len(documents)} documents from {docx_file}")
        except Exception as e:
            print(f"Error loading {docx_file}: {e}")

    json_files = list(data_path.glob("**/*.json"))
    print(f"Found {len(json_files)} JSON files.")
    for json_file in json_files:
        try:
            loader = JSONLoader(str(json_file))
            documents.extend(loader.load())
            print(f"Loaded {len(documents)} documents from {json_file}")
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    excel_files = list(data_path.glob("**/*.xlsx"))
    print(f"Found {len(excel_files)} Excel files.")
    for excel_file in excel_files:
        try:
            loader = UnstructuredExcelLoader(str(excel_file))
            documents.extend(loader.load())
            print(f"Loaded {len(documents)} documents from {excel_file}")
        except Exception as e:
            print(f"Error loading {excel_file}: {e}")   

    return documents