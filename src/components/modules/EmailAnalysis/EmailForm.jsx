import React, { useState } from 'react';
import Input from '../../common/Input';
import Button from '../../common/Button';

const EmailForm = ({ onSubmit, loading }) => {
    const [email, setEmail] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (email.trim()) {
            onSubmit(email.trim());
        }
    };

    return (
        <form onSubmit={handleSubmit} className="analysis-form">
            <h2>Enter Email Address</h2>
            <Input
                label="Email Address"
                type="email"
                placeholder="example@email.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                hint="We'll check if this email has appeared in any known data breaches"
                required
            />
            <Button
                type="submit"
                variant="primary"
                size="lg"
                block
                loading={loading}
                disabled={!email.trim()}
            >
                Analyze Email
            </Button>
        </form>
    );
};

export default EmailForm;
