import React from 'react';
import { Navigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const ProtectedRoute = ({ children }) => {
    const { user, loading } = useAuth();

    if (loading) {
        return (
            <div className="auth-page">
                <div className="auth-glow"></div>
                <div className="auth-container" style={{ textAlign: 'center' }}>
                    <div className="auth-spinner" style={{
                        width: 40,
                        height: 40,
                        borderColor: 'rgba(0, 235, 135, 0.2)',
                        borderTopColor: '#00eb87',
                        margin: '0 auto'
                    }}></div>
                    <p style={{ color: '#8b9a8f', marginTop: 16 }}>Verifying clearance...</p>
                </div>
            </div>
        );
    }

    if (!user) {
        return <Navigate to="/login" replace />;
    }

    return children;
};

export default ProtectedRoute;
