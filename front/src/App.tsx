import React, { useState } from 'react';
import { IconSpan } from '@hideo54/reactor';
import { CloudUpload, Download } from '@styled-icons/ionicons-outline';
import './App.css';

function App() {
  const [isLoading, setIsLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [depthImageUrl, setDepthImageUrl] = useState<string | null>(null);
  const [depth, setDepth] = useState<number[][] | null>(null);

  const onFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    setErrorMessage(null);
    if (files && files.length > 0) {
      const file = files[0];
      const formData = new FormData();
      formData.append('file', file);
      setIsLoading(true);
      setDepthImageUrl(null);
      setDepth(null);
      const res = await fetch('http://localhost:8080', {
        method: 'POST',
        body: formData,
      });
      if (res.status === 200) {
        const { depth, filename } = await res.json();
        setDepthImageUrl(`https://storage.googleapis.com/depth-web/${filename}`);
        setDepth(depth);
      } else {
        setErrorMessage('エラーが発生しました');
      }
      setIsLoading(false);
    } else {
      setErrorMessage('適切なファイルが選択されていません');
    }
  };
  return (
    <div className="App">
      <h1>DepthWeb</h1>
      {isLoading ? (
        <p>Loading...</p>
      ) : (
        <label className='upload'>
          <IconSpan LeftIcon={CloudUpload}>画像をアップロード</IconSpan>
          <input type='file' name='file' accept='image/png, image/jpeg' required onChange={onFileUpload} />
        </label>
      )}
      {errorMessage && <p>{errorMessage}</p>}
      {depthImageUrl &&
        <section>
          <h2>深度画像</h2>
          <img src={depthImageUrl} alt='Depth' />
          <h2>点群</h2>
          <p>TODO:</p>
        </section>
      }
    </div>
  );
}

export default App;
