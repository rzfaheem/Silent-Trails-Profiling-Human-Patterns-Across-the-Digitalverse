import React from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';

export const ElitePlanCard = ({
    className,
    imageUrl,
    title,
    subtitle,
    description,
    highlights = [],
    link,
    ...props
}) => {
    return (
        <Link to={link || "#"} className="elite-card-link" style={{ textDecoration: 'none', display: 'block', height: '100%' }}>
            <motion.div
                whileHover={{ scale: 1.02 }}
                transition={{ type: "spring", stiffness: 250, damping: 20 }}
                className={`elite-card ${className || ''}`}
                {...props}
            >
                {/* Image Section */}
                <motion.div
                    className="elite-card-image-container"
                    whileHover={{ scale: 1.1 }}
                    transition={{ duration: 0.45 }}
                >
                    <img
                        src={imageUrl}
                        alt={title}
                        className="elite-card-image"
                    />
                    <div className="elite-card-gradient" />
                </motion.div>

                {/* Content Section */}
                <div className="elite-card-content">
                    <p className="elite-card-subtitle">
                        {subtitle}
                    </p>
                    <h3 className="elite-card-title">{title}</h3>
                    <p className="elite-card-description">
                        {description}
                    </p>

                    {/* Highlights */}
                    {highlights.length > 0 && (
                        <ul className="elite-card-highlights">
                            {highlights.map((item, idx) => (
                                <li key={idx} className="elite-card-highlight-item">
                                    <span className="elite-highlight-dot" />
                                    {item}
                                </li>
                            ))}
                        </ul>
                    )}

                    {/* Footer / CTA */}
                    <div className="elite-card-footer">
                        <div className="elite-card-button">
                            Access Module
                        </div>
                    </div>
                </div>
            </motion.div>
        </Link>
    );
};
