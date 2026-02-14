import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Layout from './components/layout/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import Home from './pages/Home';
import Login from './pages/Login';
import Signup from './pages/Signup';
import SocialMapping from './pages/SocialMapping'; // Digital Recon
import DeepfakeForensics from './pages/DeepfakeForensics';
import Timeline from './pages/Timeline';
import './styles/dashboard.css';

function App() {
  return (
    <Router>
      <Routes>
        {/* Public routes - no layout */}
        <Route path="/login" element={<Login />} />
        <Route path="/signup" element={<Signup />} />

        {/* Protected routes - with layout */}
        <Route path="/" element={
          <ProtectedRoute>
            <Layout><Home /></Layout>
          </ProtectedRoute>
        } />
        <Route path="/digital-recon" element={
          <ProtectedRoute>
            <Layout><SocialMapping /></Layout>
          </ProtectedRoute>
        } />
        <Route path="/deepfake-forensics" element={
          <ProtectedRoute>
            <Layout><DeepfakeForensics /></Layout>
          </ProtectedRoute>
        } />
        <Route path="/timeline" element={
          <ProtectedRoute>
            <Layout><Timeline /></Layout>
          </ProtectedRoute>
        } />
      </Routes>
    </Router>
  );
}

export default App;
