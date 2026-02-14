import React from 'react';

const Loader = ({ size = 40, color = '#a855f7', message = 'Loading...' }) => {
    return (
        <div className="loader-container" style={{
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            padding: '2rem',
            gap: '1rem'
        }}>
            <span
                className="loader-spinner"
                style={{
                    width: size,
                    height: size,
                    borderWidth: 3,
                    borderColor: `${color}33`,
                    borderTopColor: color
                }}
            ></span>
            {message && <span style={{ color: '#94a3b8', fontSize: '0.9rem' }}>{message}</span>}
        </div>
    );
};

export default Loader;
