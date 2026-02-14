import React from 'react';
import { Link, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const Header = () => {
    const { user, signOut } = useAuth();
    const navigate = useNavigate();

    const handleSignOut = async () => {
        await signOut();
        navigate('/login');
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
                        <button onClick={handleSignOut} className="header-signout">
                            Sign Out
                        </button>
                    )}
                </nav>
            </div>
        </header>
    );
};

export default Header;
