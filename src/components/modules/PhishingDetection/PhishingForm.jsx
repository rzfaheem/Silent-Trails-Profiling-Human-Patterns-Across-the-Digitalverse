import React, { useState } from 'react';
import Input from '../../common/Input';
import Button from '../../common/Button';

const PhishingForm = ({ onSubmit, loading }) => {
    const [input, setInput] = useState('');
    const [inputType, setInputType] = useState('url');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (input.trim()) {
            onSubmit(input.trim());
        }
    };

    return (
        <form onSubmit={handleSubmit} className="analysis-form">
            <h2>Analyze for Phishing</h2>

            <div style={{ marginBottom: 'var(--spacing-lg)' }}>
                <label className="form-label">Input Type</label>
                <div style={{ display: 'flex', gap: 'var(--spacing-md)' }}>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-xs)', cursor: 'pointer' }}>
                        <input
                            type="radio"
                            name="inputType"
                            value="url"
                            checked={inputType === 'url'}
                            onChange={(e) => setInputType(e.target.value)}
                        />
                        URL
                    </label>
                    <label style={{ display: 'flex', alignItems: 'center', gap: 'var(--spacing-xs)', cursor: 'pointer' }}>
                        <input
                            type="radio"
                            name="inputType"
                            value="message"
                            checked={inputType === 'message'}
                            onChange={(e) => setInputType(e.target.value)}
                        />
                        Message/Text
                    </label>
                </div>
            </div>

            {inputType === 'url' ? (
                <Input
                    label="Suspicious URL"
                    type="url"
                    placeholder="https://suspicious-link.com/login"
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    hint="Enter the URL you want to analyze for phishing indicators"
                    required
                />
            ) : (
                <Input
                    label="Suspicious Message"
                    textarea
                    rows={5}
                    placeholder="Paste the suspicious email or message content here..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    hint="Enter the message text you want to analyze for phishing patterns"
                    required
                />
            )}

            <Button
                type="submit"
                variant="primary"
                size="lg"
                block
                loading={loading}
                disabled={!input.trim()}
            >
                Detect Phishing
            </Button>
        </form>
    );
};

export default PhishingForm;
