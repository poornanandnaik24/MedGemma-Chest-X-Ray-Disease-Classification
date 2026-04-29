import { useState, useRef } from 'react';
import { Client } from "@gradio/client";
import { Upload, Stethoscope, MessageSquare, Loader2 } from 'lucide-react';
import './index.css';

function App() {
  const [image, setImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [question, setQuestion] = useState('');
  const [result, setResult] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('classify');
  const fileInputRef = useRef(null);

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      setImage(file);
      const url = URL.createObjectURL(file);
      setImagePreview(url);
      setResult('');
    }
  };

  const handleClassify = async () => {
    if (!image) {
      setResult("Please upload an image first.");
      return;
    }
    
    setIsLoading(true);
    setResult('');
    
    try {
      const app = await Client.connect("poornanandnaik24/MedGemma-ChestXRay-Classifier");
      // Gradio predict endpoint usually maps to /predict for the first button click event
      const response = await app.predict("/predict", [image]);
      setResult(response.data[0]);
    } catch (error) {
      setResult("Error connecting to the model: " + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAskQuestion = async () => {
    if (!image) {
      setResult("Please upload an image first.");
      return;
    }
    if (!question.trim()) {
      setResult("Please ask a question.");
      return;
    }
    
    setIsLoading(true);
    setResult('');
    
    try {
      const app = await Client.connect("poornanandnaik24/MedGemma-ChestXRay-Classifier");
      // Q&A is the second click event, usually maps to /predict_1
      const response = await app.predict("/predict_1", [image, question]);
      setResult(response.data[0]);
    } catch (error) {
      setResult("Error connecting to the model: " + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="app-wrapper">
      <header className="header">
        <h1>MedGemma AI</h1>
        <p>Advanced Chest X-Ray Disease Classification & Analysis</p>
      </header>

      <main className="glass-panel main-content">
        <section className="upload-section">
          <input 
            type="file" 
            ref={fileInputRef} 
            onChange={handleImageUpload} 
            accept="image/*" 
            className="hidden-input"
          />
          <div className="upload-box" onClick={() => fileInputRef.current?.click()}>
            {imagePreview ? (
              <img src={imagePreview} alt="Chest X-Ray preview" />
            ) : (
              <div className="upload-placeholder">
                <Upload size={48} />
                <span>Click to upload Chest X-Ray</span>
                <span style={{ fontSize: '0.8rem', opacity: 0.6 }}>JPEG, PNG supported</span>
              </div>
            )}
          </div>
        </section>

        <section className="controls-section">
          <div className="tabs">
            <button 
              className={`tab-btn ${activeTab === 'classify' ? 'active' : ''}`}
              onClick={() => setActiveTab('classify')}
            >
              <Stethoscope size={20} />
              Classify Disease
            </button>
            <button 
              className={`tab-btn ${activeTab === 'qa' ? 'active' : ''}`}
              onClick={() => setActiveTab('qa')}
            >
              <MessageSquare size={20} />
              Custom Q&A
            </button>
          </div>

          <div className="input-area">
            {activeTab === 'classify' ? (
              <>
                <div style={{ color: 'var(--text-muted)', fontSize: '0.9rem', marginBottom: '1rem', background: 'rgba(0,0,0,0.2)', padding: '1rem', borderRadius: '8px' }}>
                  <strong style={{ color: 'var(--text-main)' }}>Auto-applied Prompt:</strong><br />
                  What is the disease?<br />
                  A: Covid19<br />
                  B: Normal<br />
                  C: Pneumonia<br />
                  D: Tuberculosis
                </div>
                <button 
                  className="action-btn" 
                  onClick={handleClassify}
                  disabled={isLoading || !image}
                >
                  {isLoading ? <Loader2 className="spin" /> : <Stethoscope />}
                  {isLoading ? 'Analyzing...' : 'Run Classification'}
                </button>
              </>
            ) : (
              <>
                <textarea 
                  className="text-input"
                  placeholder="E.g., What are the visible abnormalities?"
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                />
                <button 
                  className="action-btn" 
                  onClick={handleAskQuestion}
                  disabled={isLoading || !image || !question.trim()}
                >
                  {isLoading ? <Loader2 className="spin" /> : <MessageSquare />}
                  {isLoading ? 'Generating Answer...' : 'Ask Question'}
                </button>
              </>
            )}
          </div>

          <div className="result-area">
            <h3>Diagnosis / Result</h3>
            <div className="result-text">
              {result || <span style={{ color: 'var(--text-muted)' }}>Awaiting image analysis...</span>}
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;
