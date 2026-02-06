import React from 'react';
import { FaSeedling, FaGoogle, FaFacebook } from 'react-icons/fa';

/**
 * Login Page Component
 * @param {Function} setIsLoggedIn - Updates authentication state
 * @param {Function} setUserName - Sets the display name for the dashboard
 * @param {Function} onBack - Navigates back to the Landing page
 * @param {string} inputName - State for the name input
 * @param {Function} setInputName - Setter for the name input
 * @param {string} email - State for the email input
 * @param {Function} setEmail - Setter for the email input
 * @param {string} password - State for the password input
 * @param {Function} setPassword - Setter for the password input
 */
const Login = ({ 
  setIsLoggedIn, 
  setUserName, 
  onBack, 
  inputName, 
  setInputName, 
  email, 
  setEmail, 
  password, 
  setPassword 
}) => {

  const handleLogin = (e) => {
    e.preventDefault();
    if (inputName && email && password) {
      setUserName(inputName);
      setIsLoggedIn(true);
    }
  };

  const handleSocialLogin = (platform) => {
    setUserName(`${platform} User`);
    setIsLoggedIn(true);
  };

  return (
    <div className="login-container">
      <div className="login-card">
        <div style={{ color: '#AEB877', fontSize: '3.5rem', marginBottom: '10px' }}>
          <FaSeedling />
        </div>
        <h2 style={{ color: '#1f2937', marginBottom: '5px' }}>Welcome to FarmGuard</h2>
        <p style={{ color: '#64748b', marginBottom: '30px' }}>Smart AI for Smarter Farming</p>
        
        <form onSubmit={handleLogin}>
          <div className="login-input-group">
            <label className="login-label">Full Name</label>
            <input 
              type="text" 
              className="login-input" 
              placeholder="e.g. Rajesh Kumar" 
              value={inputName} 
              onChange={(e) => setInputName(e.target.value)} 
              required 
            />
          </div>
          
          <div className="login-input-group">
            <label className="login-label">Email Address</label>
            <input 
              type="email" 
              className="login-input" 
              placeholder="name@example.com" 
              value={email} 
              onChange={(e) => setEmail(e.target.value)} 
              required 
            />
          </div>
          
          <div className="login-input-group">
            <label className="login-label">Password</label>
            <input 
              type="password" 
              className="login-input" 
              placeholder="••••••••" 
              value={password} 
              onChange={(e) => setPassword(e.target.value)} 
              required 
            />
          </div>
          
          <button type="submit" className="login-btn">Login to Dashboard</button>
        </form>
        
        <div className="divider">OR CONTINUE WITH</div>
        
        <div className="social-login">
          <button className="social-btn" onClick={() => handleSocialLogin('Google')}>
            <FaGoogle style={{ color: '#DB4437' }} /> Google
          </button>
          <button className="social-btn" onClick={() => handleSocialLogin('Facebook')}>
            <FaFacebook style={{ color: '#4267B2' }} /> Facebook
          </button>
        </div>
        
        <div 
          style={{ marginTop: '20px', fontSize: '0.9rem', color: '#64748b', cursor: 'pointer' }} 
          onClick={onBack}
        >
          ← Back to Home
        </div>
      </div>
    </div>
  );
};

export default Login;