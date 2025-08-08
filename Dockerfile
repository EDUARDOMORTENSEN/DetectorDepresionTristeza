# Usa la imagen oficial de Python 3.10 slim
FROM python:3.10-slim

# Directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia el archivo requirements.txt al contenedor
COPY requirements.txt .

# Instala pip actualizado y las dependencias
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copia todos los archivos del proyecto al contenedor
COPY . .

# Expone el puerto donde Streamlit corre por defecto
EXPOSE 8501

# Comando para ejecutar tu app con Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
