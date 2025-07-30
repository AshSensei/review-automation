import Home from './pages/Home.tsx';
import HTMLAnalyzer from './pages/HTMLAnalyzer.tsx';
import { Routes, Route } from 'react-router-dom';
import CompetitorAnalyzer from './pages/CompetitorAnalyzer.tsx';
import FeatureAnalyzer from './pages/FeatureAnalyzer.tsx';
import './App.css'

function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/html-analyzer" element={<HTMLAnalyzer />} />
      <Route path="/competitor-analyzer" element={<CompetitorAnalyzer />} />
      <Route path="/feature-analyzer" element={<FeatureAnalyzer />} />
    </Routes>
  );
}

export default App
