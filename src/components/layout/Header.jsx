import React, { useState } from 'react';
import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const Header = () => {
    const { user, signOut } = useAuth();
    const navigate = useNavigate();
    const [theme, setTheme] = useState(localStorage.getItem('st_theme') || 'dark');

    const handleSignOut = async () => {
        await signOut();
        navigate('/login');
    };

    const toggleTheme = () => {
        const newTheme = theme === 'dark' ? 'light' : 'dark';
        setTheme(newTheme);
        localStorage.setItem('st_theme', newTheme);
        document.documentElement.setAttribute('data-theme', newTheme);
    };

    return (
        <header className="header">
            <div className="header-container">
                <Link to="/" className="header-logo">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M12 2L2 7l10 5 10-5-10-5z" />
                        <path d="M2 17l10 5 10-5" />
                        <path d="M2 12l10 5 10-5" />
                    </svg>
                    <span>Silent Trails</span>
                </Link>

                <nav className="header-nav">
                    <NavLink to="/" end>
                        Home
                    </NavLink>
                    <NavLink to="/digital-recon">
                        Digital Recon
                    </NavLink>
                    <NavLink to="/deepfake-forensics">
                        Deepfake
                    </NavLink>
                    <NavLink to="/timeline">
                        Timeline
                    </NavLink>
                    {user && (
                        <>
                            <button onClick={toggleTheme} className="header-theme-toggle" style={{ background: 'transparent', border: '1px solid var(--border)', color: 'var(--text-primary)', padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', marginLeft: '16px', marginRight: '8px' }}>
                                {theme === 'dark' ? '☀️ Light' : '🌙 Dark'}
                            </button>
                            <button onClick={handleSignOut} className="header-signout">
                                Sign Out
                            </button>
                        </>
                    )}
                </nav>
            </div>
        </header>
    );
};

export default Header;
