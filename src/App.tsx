import { Container, Navbar } from 'react-bootstrap'
import StimulusDisplay from './components/StimuliPresentation/StimulusDisplay'

function App() {
  return (
    <div className="App">
      <Navbar bg="dark" variant="dark" expand="lg">
        <Container>
          <Navbar.Brand>EEG Stimulus Presentation</Navbar.Brand>
        </Container>
      </Navbar>

      <Container fluid className="mt-3">
        <h2>Stimulus Presentation</h2>
        <StimulusDisplay />
      </Container>
    </div>
  )
}

export default App