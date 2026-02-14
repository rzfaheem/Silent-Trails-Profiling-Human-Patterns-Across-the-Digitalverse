import React from 'react';

const Footer = () => {
    const currentYear = new Date().getFullYear();

    return (
        <footer className="footer">
            <div className="footer-container">
                <p className="footer-title">Silent Trails</p>
                <p className="footer-subtitle">
                    Profiling Human Patterns Across the Digitalverse
                </p>
                <p className="footer-subtitle">
                    Final Year Project - Academic Prototype
                </p>
                <p className="footer-copyright">
                    © {currentYear} Silent Trails. For educational purposes only.
                </p>
            </div>
        </footer>
    );
};

export default Footer;
