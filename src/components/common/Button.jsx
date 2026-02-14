import React from 'react';

const Button = ({
    children,
    variant = 'primary',
    size = 'md',
    block = false,
    disabled = false,
    loading = false,
    type = 'button',
    onClick,
    className = '',
    ...props
}) => {
    const classes = [
        'btn',
        `btn-${variant}`,
        size === 'lg' && 'btn-lg',
        block && 'btn-block',
        className
    ].filter(Boolean).join(' ');

    return (
        <button
            type={type}
            className={classes}
            disabled={disabled || loading}
            onClick={onClick}
            {...props}
        >
            {loading ? (
                <>
                    <span className="loader-spinner" style={{ width: 20, height: 20, borderWidth: 2 }}></span>
                    <span>Analyzing...</span>
                </>
            ) : children}
        </button>
    );
};

export default Button;
