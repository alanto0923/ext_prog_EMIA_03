// frontend/next.config.js
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Optional: If serving images from the backend output dir directly
//   images: {
//     remotePatterns: [
//       {
//         protocol: 'http', // or 'https' if backend is served over HTTPS
//         hostname: 'localhost', // or your backend domain
//         port: '8000', // backend port
//         pathname: '/static/output/**', // Allow images from the output directory
//       },
//     ],
//   },
};

module.exports = nextConfig;