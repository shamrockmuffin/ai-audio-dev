import streamlit as st
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class SessionState:
    """Wrapper for Streamlit session state with additional functionality"""
    
    def __init__(self):
        self._initialize_defaults()
    
    def _initialize_defaults(self):
        """Initialize default session state values"""
        defaults = {
            'current_file': None,
            'enhanced_file': None,
            'file_history': [],
            'transcription': None,
            'enhancement_result': None,
            'chat_history': [],
            'settings': {
                'noise_reduction': True,
                'normalize': True,
                'enhance_transcription': True,
                'band_pass': True,
                'compression': False,
                'target_db': -20.0,
                'compression_ratio': 4.0,
                'noise_reduction_strength': 1.0
            },
            'ui_state': {
                'active_tab': 0,
                'show_advanced': False,
                'theme': 'light'
            }
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def __getattr__(self, name):
        """Get attribute from session state"""
        if name in st.session_state:
            return st.session_state[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
    
    def __setattr__(self, name, value):
        """Set attribute in session state"""
        if name.startswith('_'):
            super().__setattr__(name, value)
        else:
            st.session_state[name] = value
    
    def add_file(self, file_path: str, file_name: str):
        """Add file to history"""
        file_info = {
            'path': file_path,
            'name': file_name,
            'timestamp': datetime.now().isoformat(),
            'size': Path(file_path).stat().st_size
        }
        
        # Add to history
        st.session_state.file_history.append(file_info)
        
        # Set as current
        st.session_state.current_file = file_path
        
        # Limit history size
        if len(st.session_state.file_history) > 20:
            st.session_state.file_history = st.session_state.file_history[-20:]
    
    def remove_file(self, index: int):
        """Remove file from history"""
        if 0 <= index < len(st.session_state.file_history):
            removed = st.session_state.file_history.pop(index)
            
            # If it was the current file, clear it
            if st.session_state.current_file == removed['path']:
                st.session_state.current_file = None
                st.session_state.enhanced_file = None
                st.session_state.transcription = None
    
    def update_settings(self, settings: Dict[str, Any]):
        """Update settings"""
        st.session_state.settings.update(settings)
    
    def add_chat_message(self, role: str, content: str):
        """Add message to chat history"""
        message = {
            'role': role,
            'content': content,
            'timestamp': datetime.now().isoformat()
        }
        st.session_state.chat_history.append(message)
        
        # Limit chat history
        if len(st.session_state.chat_history) > 100:
            st.session_state.chat_history = st.session_state.chat_history[-100:]
    
    def clear_chat_history(self):
        """Clear chat history"""
        st.session_state.chat_history = []
    
    def export_session(self) -> Dict:
        """Export session state for saving"""
        return {
            'settings': st.session_state.settings,
            'file_history': st.session_state.file_history,
            'chat_history': st.session_state.chat_history,
            'timestamp': datetime.now().isoformat()
        }
    
    def import_session(self, data: Dict):
        """Import session state from saved data"""
        if 'settings' in data:
            st.session_state.settings.update(data['settings'])
        if 'file_history' in data:
            st.session_state.file_history = data['file_history']
        if 'chat_history' in data:
            st.session_state.chat_history = data['chat_history']
    
    def reset(self):
        """Reset session state to defaults"""
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        self._initialize_defaults()

class SessionStateManager:
    """Manager for session state with persistence capabilities"""
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path or Path.home() / '.audio_analyzer' / 'sessions'
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.state = SessionState()
    
    def get_state(self) -> SessionState:
        """Get session state instance"""
        return self.state
    
    def save_session(self, name: str):
        """Save current session to file"""
        try:
            session_data = self.state.export_session()
            file_path = self.storage_path / f"{name}.json"
            
            with open(file_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            logger.info(f"Session saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving session: {e}")
            return False
    
    def load_session(self, name: str):
        """Load session from file"""
        try:
            file_path = self.storage_path / f"{name}.json"
            
            if not file_path.exists():
                logger.warning(f"Session file not found: {file_path}")
                return False
            
            with open(file_path, 'r') as f:
                session_data = json.load(f)
            
            self.state.import_session(session_data)
            logger.info(f"Session loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading session: {e}")
            return False
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List available saved sessions"""
        sessions = []
        
        for file_path in self.storage_path.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                sessions.append({
                    'name': file_path.stem,
                    'timestamp': data.get('timestamp', ''),
                    'file_count': len(data.get('file_history', [])),
                    'path': str(file_path)
                })
            except:
                continue
        
        return sorted(sessions, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_session(self, name: str):
        """Delete saved session"""
        try:
            file_path = self.storage_path / f"{name}.json"
            if file_path.exists():
                file_path.unlink()
                logger.info(f"Session deleted: {name}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Error deleting session: {e}")
            return False 