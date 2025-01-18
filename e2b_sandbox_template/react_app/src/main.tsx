import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <div className="w-full h-full p-0 m-0">
      <App />
    </div>
  </StrictMode>,
)