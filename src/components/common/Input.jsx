import React from 'react';

const Input = ({
    label,
    type = 'text',
    name,
    value,
    onChange,
    placeholder = '',
    required = false,
    disabled = false,
    className = '',
    ...props
}) => {
    return (
        <div className={`input-group ${className}`}>
            {label && <label htmlFor={name} className="input-label">{label}</label>}
            <input
                id={name}
                type={type}
                name={name}
                value={value}
                onChange={onChange}
                placeholder={placeholder}
                required={required}
                disabled={disabled}
                className="input-field"
                {...props}
            />
        </div>
    );
};

export default Input;
