import React, { createContext, useContext, useEffect, useState } from 'react';

const AuthContext = createContext({});

export const useAuth = () => useContext(AuthContext);

export const AuthProvider = ({ children }) => {
    const [user, setUser] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        // Mock initial session check
        setTimeout(() => {
            // Check local storage for mock user session to persist login
            const storedUser = localStorage.getItem('mock_user');
            if (storedUser) {
                setUser(JSON.parse(storedUser));
            } else {
                setUser(null);
            }
            setLoading(false);
        }, 500);
    }, []);

    const signUp = async (email, password, fullName) => {
        return new Promise((resolve) => {
            setTimeout(() => {
                const mockUser = { id: 'mock-123', email, user_metadata: { full_name: fullName } };
                setUser(mockUser);
                localStorage.setItem('mock_user', JSON.stringify(mockUser));
                resolve({ data: { user: mockUser }, error: null });
            }, 1000);
        });
    };

    const signIn = async (email, password) => {
        return new Promise((resolve) => {
            setTimeout(() => {
                const mockUser = { id: 'mock-123', email, user_metadata: { full_name: 'Mock User' } };
                setUser(mockUser);
                localStorage.setItem('mock_user', JSON.stringify(mockUser));
                resolve({ data: { user: mockUser }, error: null });
            }, 1000);
        });
    };

    const signOut = async () => {
        return new Promise((resolve) => {
            setTimeout(() => {
                setUser(null);
                localStorage.removeItem('mock_user');
                resolve({ error: null });
            }, 500);
        });
    };

    const value = {
        user,
        loading,
        signUp,
        signIn,
        signOut
    };

    return (
        <AuthContext.Provider value={value}>
            {children}
        </AuthContext.Provider>
    );
};
