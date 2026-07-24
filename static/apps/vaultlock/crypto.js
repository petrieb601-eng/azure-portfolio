/**
 * SecureVault — crypto.js
 * All cryptographic operations using the Web Crypto API
 */

const CryptoEngine = (() => {

  const enc = new TextEncoder();
  const dec = new TextDecoder();

  /* ---- KEY DERIVATION ---- */
  async function deriveKey(passphrase, salt, bits = 256) {
    const keyMaterial = await crypto.subtle.importKey(
      'raw', enc.encode(passphrase), 'PBKDF2', false, ['deriveKey']
    );
    return crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt, iterations: 100000, hash: 'SHA-256' },
      keyMaterial,
      { name: 'AES-GCM', length: bits },
      false,
      ['encrypt', 'decrypt']
    );
  }

  /* ---- AES-GCM ENCRYPT ---- */
  async function encryptAESGCM(plaintext, passphrase) {
    const salt = crypto.getRandomValues(new Uint8Array(16));
    const iv   = crypto.getRandomValues(new Uint8Array(12));
    const key  = await deriveKey(passphrase, salt);
    const encrypted = await crypto.subtle.encrypt(
      { name: 'AES-GCM', iv },
      key,
      enc.encode(plaintext)
    );
    const combined = new Uint8Array(salt.length + iv.length + encrypted.byteLength);
    combined.set(salt, 0);
    combined.set(iv,   salt.length);
    combined.set(new Uint8Array(encrypted), salt.length + iv.length);
    return btoa(String.fromCharCode(...combined));
  }

  /* ---- AES-GCM DECRYPT ---- */
  async function decryptAESGCM(ciphertext, passphrase) {
    const combined = Uint8Array.from(atob(ciphertext), c => c.charCodeAt(0));
    const salt = combined.slice(0, 16);
    const iv   = combined.slice(16, 28);
    const data = combined.slice(28);
    const key  = await deriveKey(passphrase, salt);
    const decrypted = await crypto.subtle.decrypt(
      { name: 'AES-GCM', iv },
      key,
      data
    );
    return dec.decode(decrypted);
  }

  /* ---- XOR ---- */
  function xorCipher(text, key) {
    let result = '';
    for (let i = 0; i < text.length; i++) {
      result += String.fromCharCode(text.charCodeAt(i) ^ key.charCodeAt(i % key.length));
    }
    return btoa(unescape(encodeURIComponent(result)));
  }

  function xorDecipher(b64, key) {
    const text = decodeURIComponent(escape(atob(b64)));
    let result = '';
    for (let i = 0; i < text.length; i++) {
      result += String.fromCharCode(text.charCodeAt(i) ^ key.charCodeAt(i % key.length));
    }
    return result;
  }

  /* ---- SHA HASH ---- */
  async function hashText(text, algo = 'SHA-256') {
    const hashBuffer = await crypto.subtle.digest(algo, enc.encode(text));
    return Array.from(new Uint8Array(hashBuffer))
      .map(b => b.toString(16).padStart(2, '0'))
      .join('');
  }

  /* ---- KEY GENERATION ---- */
  function generateKey(bits = 256, format = 'hex') {
    const bytes = bits / 8;
    const arr = crypto.getRandomValues(new Uint8Array(bytes));
    if (format === 'hex')    return Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join('');
    if (format === 'base64') return btoa(String.fromCharCode(...arr));
    if (format === 'binary') return Array.from(arr).map(b => b.toString(2).padStart(8, '0')).join(' ');
    return Array.from(arr).map(b => b.toString(16).padStart(2, '0')).join('');
  }

  function generatePassphrase() {
    const words = [
      'amber','bridge','cipher','delta','echo','falcon','gravity','harbor',
      'indigo','jungle','krypton','lunar','matrix','nebula','orbit','phantom',
      'quartz','raven','shadow','tiger','ultra','vortex','whisper','xenon',
      'yellow','zephyr','anchor','blaze','crystal','drift'
    ];
    const picked = [];
    const arr = crypto.getRandomValues(new Uint8Array(4));
    for (let i = 0; i < 4; i++) {
      picked.push(words[arr[i] % words.length]);
    }
    const num = crypto.getRandomValues(new Uint8Array(1))[0] % 900 + 100;
    return picked.join('-') + '-' + num;
  }

  function generateUUID() {
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
      const r = crypto.getRandomValues(new Uint8Array(1))[0] % 16;
      return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
  }

  /* ---- KEY STRENGTH ---- */
  function keyStrength(key) {
    let score = 0;
    if (key.length >= 8)  score += 20;
    if (key.length >= 16) score += 20;
    if (key.length >= 32) score += 10;
    if (/[A-Z]/.test(key))      score += 15;
    if (/[a-z]/.test(key))      score += 10;
    if (/[0-9]/.test(key))      score += 15;
    if (/[^A-Za-z0-9]/.test(key)) score += 10;
    const labels = ['', 'Weak', 'Fair', 'Good', 'Strong', 'Excellent'];
    const colors = ['#606068', '#f76e6e', '#f5c542', '#f5c542', '#7ee8a2', '#5bc4f5'];
    const idx = score === 0 ? 0 : score <= 20 ? 1 : score <= 40 ? 2 : score <= 60 ? 3 : score <= 80 ? 4 : 5;
    return { score: Math.min(score, 100), label: labels[idx], color: colors[idx] };
  }

  return {
    encryptAESGCM,
    decryptAESGCM,
    xorCipher,
    xorDecipher,
    hashText,
    generateKey,
    generatePassphrase,
    generateUUID,
    keyStrength
  };
})();
