/**
 * Electron Native Desktop Wrapper for Fraud Ring Finder.
 * Launches the Graph ML Forensic Workstation inside a native desktop window on Windows, macOS, and Linux.
 *
 * Run with:
 *   npm run desktop
 */

const { app, BrowserWindow, shell } = require('electron');
const { spawn } = require('child_process');
const http = require('http');
const path = require('path');

let mainWindow = null;
let serverProcess = null;
const PORT = process.env.PORT || 3000;
const APP_URL = `http://localhost:${PORT}`;

function waitForServer(url, timeoutMs = 15000) {
  const start = Date.now();
  return new Promise((resolve, reject) => {
    const check = () => {
      http
        .get(url, (res) => {
          if (res.statusCode >= 200 && res.statusCode < 500) {
            resolve();
          } else if (Date.now() - start > timeoutMs) {
            reject(new Error('Server startup timed out'));
          } else {
            setTimeout(check, 200);
          }
        })
        .on('error', () => {
          if (Date.now() - start > timeoutMs) {
            reject(new Error('Server startup timed out'));
          } else {
            setTimeout(check, 200);
          }
        });
    };
    check();
  });
}

async function createWindow() {
  serverProcess = spawn('npx', ['tsx', 'server.ts'], {
    cwd: path.resolve(__dirname, '..'),
    env: { ...process.env, PORT: String(PORT) },
    stdio: 'inherit',
    shell: true,
  });

  await waitForServer(APP_URL);

  mainWindow = new BrowserWindow({
    width: 1480,
    height: 920,
    minWidth: 1120,
    minHeight: 740,
    title: 'Fraud Ring Finder — Graph ML Forensic Workstation',
    backgroundColor: '#0f172a',
    autoHideMenuBar: true,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
    },
  });

  mainWindow.loadURL(APP_URL);

  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
  if (serverProcess) {
    serverProcess.kill();
  }
  if (process.platform !== 'darwin') {
    app.quit();
  }
});
