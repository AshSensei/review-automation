import Home from './pages/Home.tsx';
import HTMLAnalyzer from './pages/HTMLAnalyzer.tsx';
import { Routes, Route } from 'react-router-dom';
import './App.css'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/html-analyzer" element={<HTMLAnalyzer />} />
    </Routes>
  );
}

export default App
