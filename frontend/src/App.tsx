import { useState, useEffect } from 'react'
import './App.css'

interface ModelInfo {
  name: string
  version: string
  architecture: string
  threshold: number
  features: {
    input_features: number
    engineered_features: number
    total_after_encoding: number
  }
  training: {
    dataset: string
    samples: number
    optimizer: string
    loss: string
    early_stopping: string
  }
  description: string
}

interface FormData {
  age: number
  sex: number
  cp: number
  trestbps: number
  chol: number
  fbs: number
  restecg: number
  thalach: number
  exang: number
  oldpeak: number
  slope: number
  ca: number
  thal: number
}

interface PredictionResult {
  probability: number
  prediction: number
}

const initialFormData: FormData = {
  age: 55,
  sex: 1,
  cp: 0,
  trestbps: 130,
  chol: 250,
  fbs: 0,
  restecg: 0,
  thalach: 150,
  exang: 0,
  oldpeak: 1.0,
  slope: 1,
  ca: 0,
  thal: 2
}

function App() {
  const [formData, setFormData] = useState<FormData>(initialFormData)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null)
  const [loading, setLoading] = useState(false)
  const [showInfo, setShowInfo] = useState(false)

  useEffect(() => {
    fetch('http://localhost:5001/api/model-info')
      .then(res => res.json())
      .then(data => setModelInfo(data))
      .catch(() => {})
  }, [])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setLoading(true)
    try {
      const res = await fetch('http://localhost:5001/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData)
      })
      const data = await res.json()
      setResult(data)
    } catch {
      alert('Error connecting to server. Make sure the backend is running.')
    }
    setLoading(false)
  }

  const handleChange = (field: keyof FormData, value: number) => {
    setFormData(prev => ({ ...prev, [field]: value }))
    setResult(null)
  }

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="heart-icon">♥</span>
          <h1>HeartHelper</h1>
        </div>
        <p className="tagline">AI-Powered Heart Disease Risk Assessment</p>
      </header>

      <main className="main">
        <section className="form-section">
          <h2>Enter Your Health Parameters</h2>
          <form onSubmit={handleSubmit}>
            <div className="form-grid">
              <div className="form-group">
                <label>Age (years)</label>
                <input
                  type="number"
                  value={formData.age}
                  onChange={e => handleChange('age', Number(e.target.value))}
                  min={20}
                  max={100}
                />
              </div>

              <div className="form-group">
                <label>Sex</label>
                <select value={formData.sex} onChange={e => handleChange('sex', Number(e.target.value))}>
                  <option value={0}>Female</option>
                  <option value={1}>Male</option>
                </select>
              </div>

              <div className="form-group">
                <label>Chest Pain Type</label>
                <select value={formData.cp} onChange={e => handleChange('cp', Number(e.target.value))}>
                  <option value={0}>Typical Angina</option>
                  <option value={1}>Atypical Angina</option>
                  <option value={2}>Non-anginal Pain</option>
                  <option value={3}>Asymptomatic</option>
                </select>
              </div>

              <div className="form-group">
                <label>Resting Blood Pressure (mm Hg)</label>
                <input
                  type="number"
                  value={formData.trestbps}
                  onChange={e => handleChange('trestbps', Number(e.target.value))}
                  min={80}
                  max={220}
                />
              </div>

              <div className="form-group">
                <label>Cholesterol (mg/dl)</label>
                <input
                  type="number"
                  value={formData.chol}
                  onChange={e => handleChange('chol', Number(e.target.value))}
                  min={100}
                  max={600}
                />
              </div>

              <div className="form-group">
                <label>Fasting Blood Sugar &gt; 120 mg/dl</label>
                <select value={formData.fbs} onChange={e => handleChange('fbs', Number(e.target.value))}>
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>Resting ECG</label>
                <select value={formData.restecg} onChange={e => handleChange('restecg', Number(e.target.value))}>
                  <option value={0}>Normal</option>
                  <option value={1}>ST-T Wave Abnormality</option>
                  <option value={2}>Left Ventricular Hypertrophy</option>
                </select>
              </div>

              <div className="form-group">
                <label>Max Heart Rate Achieved</label>
                <input
                  type="number"
                  value={formData.thalach}
                  onChange={e => handleChange('thalach', Number(e.target.value))}
                  min={60}
                  max={220}
                />
              </div>

              <div className="form-group">
                <label>Exercise Induced Angina</label>
                <select value={formData.exang} onChange={e => handleChange('exang', Number(e.target.value))}>
                  <option value={0}>No</option>
                  <option value={1}>Yes</option>
                </select>
              </div>

              <div className="form-group">
                <label>ST Depression (oldpeak)</label>
                <input
                  type="number"
                  step="0.1"
                  value={formData.oldpeak}
                  onChange={e => handleChange('oldpeak', Number(e.target.value))}
                  min={0}
                  max={7}
                />
              </div>

              <div className="form-group">
                <label>Slope of Peak Exercise ST</label>
                <select value={formData.slope} onChange={e => handleChange('slope', Number(e.target.value))}>
                  <option value={0}>Upsloping</option>
                  <option value={1}>Flat</option>
                  <option value={2}>Downsloping</option>
                </select>
              </div>

              <div className="form-group">
                <label>Number of Major Vessels (0-3)</label>
                <select value={formData.ca} onChange={e => handleChange('ca', Number(e.target.value))}>
                  <option value={0}>0</option>
                  <option value={1}>1</option>
                  <option value={2}>2</option>
                  <option value={3}>3</option>
                </select>
              </div>

              <div className="form-group full-width">
                <label>Thalassemia</label>
                <select value={formData.thal} onChange={e => handleChange('thal', Number(e.target.value))}>
                  <option value={0}>Normal</option>
                  <option value={1}>Fixed Defect</option>
                  <option value={2}>Reversible Defect</option>
                </select>
              </div>
      </div>

            <button type="submit" className="submit-btn" disabled={loading}>
              {loading ? 'Analyzing...' : 'Analyze Risk'}
        </button>
          </form>
        </section>

        {result && (
          <section className={`result-section ${result.prediction === 1 ? 'high-risk' : 'low-risk'}`}>
            <h2>Prediction Result</h2>
            <div className="result-content">
              <div className="probability-display">
                <div className="prob-circle">
                  <svg viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="45" className="bg-circle" />
                    <circle
                      cx="50"
                      cy="50"
                      r="45"
                      className="progress-circle"
                      style={{
                        strokeDasharray: `${result.probability * 283} 283`
                      }}
                    />
                  </svg>
                  <span className="prob-value">{(result.probability * 100).toFixed(1)}%</span>
                </div>
                <p className="prob-label">Disease Probability</p>
              </div>
              <div className="prediction-text">
                <h3>{result.prediction === 1 ? 'Higher Risk Detected' : 'Lower Risk Detected'}</h3>
                <p>
                  {result.prediction === 1
                    ? 'The model indicates an elevated risk of heart disease. Please consult a healthcare professional for proper evaluation.'
                    : 'The model indicates a lower risk of heart disease. However, regular check-ups are always recommended.'}
                </p>
                <p className="threshold-note">
                  Threshold: 40% (optimized for sensitivity)
        </p>
      </div>
            </div>
          </section>
        )}

        <section className="info-section">
          <button className="info-toggle" onClick={() => setShowInfo(!showInfo)}>
            {showInfo ? 'Hide' : 'Show'} Model Information
          </button>
          
          {showInfo && modelInfo && (
            <div className="info-content">
              <h3>{modelInfo.name} v{modelInfo.version}</h3>
              <p className="description">{modelInfo.description}</p>
              
              <div className="info-grid">
                <div className="info-card">
                  <h4>Architecture</h4>
                  <p className="mono">{modelInfo.architecture}</p>
                  <p>PyTorch Neural Network with BatchNorm and Dropout</p>
                </div>
                
                <div className="info-card">
                  <h4>Features</h4>
                  <ul>
                    <li>{modelInfo.features.input_features} input features</li>
                    <li>{modelInfo.features.engineered_features} engineered features</li>
                    <li>{modelInfo.features.total_after_encoding} total (after encoding)</li>
                  </ul>
                </div>
                
                <div className="info-card">
                  <h4>Training</h4>
                  <ul>
                    <li>Dataset: {modelInfo.training.dataset}</li>
                    <li>Samples: {modelInfo.training.samples}</li>
                    <li>Loss: Weighted BCE</li>
                    <li>Early Stop: {modelInfo.training.early_stopping}</li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </section>
      </main>

      <footer className="footer">
        <p>⚠️ This tool is for educational purposes only. Not a substitute for professional medical advice.</p>
      </footer>
    </div>
  )
}

export default App
