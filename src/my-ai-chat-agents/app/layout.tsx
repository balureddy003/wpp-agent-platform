export const metadata = { title: 'Agents Chat', description: 'Chat UI with agent & tool selection' };
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body style={{ background: '#fafafa', color: '#111' }}>{children}</body>
    </html>
  );
}
