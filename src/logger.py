import sys
import os
import logging
from datetime import datetime

log_file = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
log_path = os.path.join(os.getcwd(),"logs",log_file)

log_file_path = os.path.join(log_file,log_path)

logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    
)


# if __name__ == "__main__" :
#     logging.info("logging starts")
    
    


