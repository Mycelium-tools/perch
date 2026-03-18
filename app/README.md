# Developer Guide

Set up project environment:
```
# Install pyenv and set python 3.11.9 
brew install pyenv
pyenv install 3.11.9
pyenv local 3.11.9

# Add pyenv to Path
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.zshrc
echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.zshrc
echo 'eval "$(pyenv init -)"' >> ~/.zshrc

# Check version 
python3 --version
```

Start up python backend
```
# Create the environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Start the python app
uvicorn app_api:app --reload --port 8000
```

Create your .env file (loaded by python dotenv)
```
OPENAI_API_KEY=
PINECONE_API_KEY=
PINECONE_CLOUD= # Optional
PINECONE_REGION = # Optional
```

Create your .env.local and .env.production files for use by Next.js
```
# Public keys that are passed through the browswer
NEXT_PUBLIC_SUPABASE_URL=
NEXT_PUBLIC_SUPABASE_ANON_KEY=

# Private keys that stay server side
SUPABASE_SERVICE_ROLE_KEY=
```

Start the web app at http://localhost:3000
```
cd app/src/nextjs-frontend 
npm run dev
```