import React, { useState } from 'react';
import Input from '../../common/Input';
import Button from '../../common/Button';

const SocialForm = ({ onSubmit, loading }) => {
    const [username, setUsername] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (username.trim()) {
            onSubmit(username.trim());
        }
    };

    return (
        <form onSubmit={handleSubmit} className="analysis-form">
            <h2>Search Username</h2>
            <Input
                label="Username"
                type="text"
                placeholder="john_doe"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                hint="Enter a username to search across multiple social media platforms"
                required
            />
            <Button
                type="submit"
                variant="primary"
                size="lg"
                block
                loading={loading}
                disabled={!username.trim()}
            >
                Map Social Presence
            </Button>
        </form>
    );
};

export default SocialForm;
