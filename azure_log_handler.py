import logging
from datetime import date
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceExistsError

class AzureBlobHandler(logging.Handler):
    """Handler de logging que escribe a Azure Blob Storage usando Append Blobs"""
    
    def __init__(self, connection_string, blob_name, container_name="logs"):
        super().__init__()
        self.connection_string = connection_string
        self.container_name = container_name.lower()
        self.base_blob_name = blob_name
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        
        # Crear el contenedor si no existe
        try:
            self.blob_service_client.create_container(self.container_name)
        except ResourceExistsError:
            pass
        
        # Inicializar para el día actual
        self.current_date = date.today()
        self.blob_client = self._create_blob_client()
    
    def _create_blob_client(self):
        """Crea y retorna un cliente de blob para la fecha actual"""
        blob_name = f"{self.base_blob_name}_{self.current_date}.log"
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_name
        )
        
        # Crear el blob si no existe
        try:
            blob_client.get_blob_properties()
        except:
            try:
                blob_client.create_append_blob(
                    headers={
                        "Content-Type": "text/plain; charset=utf-8",
                        "Cache-Control": "no-store"
                    }
                )
            except ResourceExistsError:
                pass
        
        return blob_client
    
    def emit(self, record):
        """Escribe el log record al append blob"""
        try:
            # Rotar blob si cambió el día
            today = date.today()
            if today != self.current_date:
                self.current_date = today
                self.blob_client = self._create_blob_client()
            
            # Formatear y escribir
            msg = self.format(record)
            if not msg.endswith('\n'):
                msg += '\n'
            
            self.blob_client.append_block(msg.encode('utf-8'))
            
        except Exception as e:
            print(f"Error escribiendo al blob: {e}")