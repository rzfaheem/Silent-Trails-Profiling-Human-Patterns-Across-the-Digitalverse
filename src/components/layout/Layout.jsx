import React from 'react';
import Header from './Header';
import Footer from './Footer';

const Layout = ({ children }) => {
    return (
        <div className="dashboard">
            <Header />
            <main className="dashboard-main">
                {children}
            </main>
            <Footer />
        </div>
    );
};

export default Layout;
