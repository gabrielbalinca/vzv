# vzv

* AI for generating offers based on user description

# Install

* Software required :
  
  - Python 3 : https://www.python.org/downloads/
  - Ollama : https://ollama.com/download 

* For installing dependencies :
```
pip install -r src/requirements.txt
ollama pull llama3
```

# Run

* Make sure all the files for training the AI are in ```train``` folder
* Question will be written in the ```tests/input.md``` and the answear will be in the ```tests/output.md```
* Run the app using this command :
```
python src/app.py
```

