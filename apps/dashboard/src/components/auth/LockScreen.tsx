'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useAuth } from '../../hooks/useAuth';
import { HudButton } from '../hud/HudButton';
import { HudInput } from '../hud/HudInput';
import { HudPanel } from '../hud/HudPanel';

export function LockScreen() {
  const router = useRouter();
  const { hasPassword, setupPassword, login } = useAuth();
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [confirm, setConfirm] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSetup = async () => {
    if (!username.trim()) {
      setError('Username is required');
      return;
    }
    if (password.length < 4) {
      setError('Password must be at least 4 characters');
      return;
    }
    if (password !== confirm) {
      setError('Passwords do not match');
      return;
    }
    setLoading(true);
    setError('');
    try {
      await setupPassword(username.trim(), password);
      router.replace('/');
    } catch {
      setError('Failed to set password. Is the server running?');
      setLoading(false);
    }
  };

  const handleLogin = async () => {
    if (!username.trim()) {
      setError('Username is required');
      return;
    }
    setLoading(true);
    setError('');
    try {
      const valid = await login(username.trim(), password);
      if (!valid) {
        setError('Invalid username or password');
        setLoading(false);
      } else {
        router.replace('/');
      }
    } catch {
      setError('Failed to connect to server');
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      if (hasPassword) handleLogin();
      else handleSetup();
    }
  };

  return (
    <div className="flex h-screen w-screen items-center justify-center bg-aiki-bg relative overflow-hidden">
      {/* Ambient gradient blobs */}
      <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[600px] h-[600px] rounded-full bg-gradient-to-br from-aiki-accent/10 to-transparent blur-3xl pointer-events-none" />
      <div
        className="absolute bottom-0 right-1/4 w-[400px] h-[400px] rounded-full bg-gradient-to-tr from-aiki-accent/[0.06] to-transparent blur-3xl pointer-events-none"
        style={{ animationDelay: '2s' }}
      />

      <div className="w-full max-w-md animate-scale-in relative z-10">
        <HudPanel title={hasPassword ? 'Authenticate' : 'Setup Password'} glow>
          <div className="flex flex-col gap-4 p-8">
            <div className="flex flex-col items-center gap-3 mb-2">
              <div
                className="w-20 h-20 rounded-full flex items-center justify-center bg-gradient-to-br from-aiki-accent/20 to-aiki-accent/10"
                style={{
                  boxShadow:
                    '0 0 40px rgba(184, 115, 51, 0.2), inset 0 0 20px rgba(184, 115, 51, 0.1)',
                }}
              >
                <span className="font-sans text-2xl font-semibold text-aiki-accent">A</span>
              </div>
              <h1 className="font-sans text-lg font-semibold text-aiki-accent">AIKIOKU</h1>
            </div>

            <HudInput
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              onKeyDown={handleKeyDown}
              autoFocus
            />

            <HudInput
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={handleKeyDown}
            />

            {!hasPassword && (
              <HudInput
                type="password"
                placeholder="Confirm password"
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                onKeyDown={handleKeyDown}
              />
            )}

            {error && <p className="text-xs text-aiki-danger">{error}</p>}

            <HudButton onClick={hasPassword ? handleLogin : handleSetup} disabled={loading}>
              {loading ? 'Verifying...' : hasPassword ? 'Unlock' : 'Set Password'}
            </HudButton>
          </div>
        </HudPanel>
      </div>
    </div>
  );
}
