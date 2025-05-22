#Back end
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
# Front-end
cd web
npm install
npm run dev