# decision_system_module.py
"""
Decision system for therapeutic animated responses
Based on evidence-based child emotional regulation and co-regulation principles
"""
import time
import random

class PlayfulResponder:
    def __init__(self, cgan_emotion_labels, exact_match_mode=False):
        """
        Initializes the Enhanced PlayfulResponder with therapeutic sequences.

        Args:
            cgan_emotion_labels (list): A list of emotion names that the cGAN understands
            exact_match_mode (bool): If True, mirror exact emotion. If False, use therapeutic sequences.
        """
        self.cgan_emotion_labels = cgan_emotion_labels
        self.emotion_to_index = {name: i for i, name in enumerate(cgan_emotion_labels)}
        self.exact_match_mode = exact_match_mode

        # Evidence-based therapeutic response sequences following co-regulation principles
        # Each sequence follows: Validation -> Gradual Regulation -> Self-Soothing -> Positive State
        self.therapeutic_sequences = {
            'neutral': [
                # Gentle encouragement sequences
                [('neutral', 40), ('happy', 20), ('neutral', 30), ('happy', 25)],  # Subtle positivity boost
                [('neutral', 35), ('surprise', 15), ('happy', 20), ('neutral', 15)]  # Gentle curiosity spark
            ],
            'happy': [
                # Amplify and sustain positive emotions (positive emotion regulation)
                [('happy', 50), ('surprise', 15), ('happy', 40), ('neutral', 10)],  # Celebrate joy
                [('happy', 30), ('happy', 30), ('surprise', 10), ('happy', 20)]  # Sustained celebration
            ],
            'sadness': [
                # Validation -> Comfort -> Hope progression (evidence-based for sadness regulation)
                [('sadness', 30), ('neutral', 20), ('happy', 25), ('neutral', 15)],  # Validate then comfort
                [('sadness', 25), ('neutral', 15), ('surprise', 10), ('happy', 20), ('neutral', 10)]  # Gentle uplift
            ],
            'surprise': [
                # Process surprise -> Settle -> Positive curiosity
                [('surprise', 25), ('neutral', 15), ('happy', 30), ('surprise', 15)],  # Process then enjoy
                [('surprise', 20), ('neutral', 10), ('happy', 25), ('neutral', 15)]  # Settle excitement
            ],
            'anger': [
                # Validation -> Breathing/Calming -> Resolution (anger regulation research)
                [('anger', 20), ('neutral', 25), ('happy', 30), ('neutral', 15)],  # Validate, calm, resolve
                [('neutral', 15), ('anger', 15), ('neutral', 30), ('happy', 20)]  # Model regulation
            ],
            'fear': [
                # Safety -> Reassurance -> Confidence building (fear regulation principles)
                [('neutral', 20), ('fear', 15), ('neutral', 25), ('happy', 25)],  # Provide safety first
                [('neutral', 25), ('surprise', 10), ('happy', 30), ('neutral', 15)]  # Build confidence
            ],
            'disgust': [
                # Acknowledge -> Redirect -> Positive engagement
                [('disgust', 15), ('neutral', 20), ('surprise', 15), ('happy', 25)],  # Quick redirect
                [('neutral', 20), ('disgust', 10), ('neutral', 20), ('happy', 20)]  # Model acceptance
            ]
        }
        
        # Exact matching sequences for direct mirroring mode
        self.exact_match_sequences = {
            emotion: [[(emotion, 60)]] for emotion in self.cgan_emotion_labels
        }

        # Default sequence if an unknown emotion is encountered
        self.default_sequence_name = 'neutral'

        # State management
        self.current_kid_emotion_name = None
        self.active_sequence = []
        self.sequence_step_index = 0
        self.frames_in_current_step = 0
        self.current_sequence_description = "None"
        self.transition_buffer = []  # For smooth transitions
        self.last_sequence_change_time = time.time()

    def set_exact_match_mode(self, exact_match):
        """Toggle between exact matching and therapeutic response modes"""
        self.exact_match_mode = exact_match
        print(f"Response mode: {'Exact Match' if exact_match else 'Therapeutic Sequences'}")

    def _get_cgan_index(self, emotion_name):
        """Get cGAN emotion index with fallback"""
        return self.emotion_to_index.get(emotion_name, self.emotion_to_index.get(self.default_sequence_name, 0))

    def _select_therapeutic_sequence(self, emotion_name):
        """Select appropriate therapeutic sequence based on research principles"""
        if emotion_name not in self.therapeutic_sequences:
            emotion_name = self.default_sequence_name
            
        # Select sequence based on recent history to avoid repetition
        available_sequences = self.therapeutic_sequences[emotion_name]
        
        # Simple selection - could be enhanced with more sophisticated logic
        selected_sequence = random.choice(available_sequences)
        
        # Convert to (index, duration) format
        return [(self._get_cgan_index(name), duration) for name, duration in selected_sequence]

    def _create_smooth_transition(self, new_sequence):
        """Create smooth transition to new sequence to avoid abrupt changes"""
        if not self.active_sequence:
            return new_sequence
            
        # Get current emotion
        current_emotion_idx = self.active_sequence[self.sequence_step_index][0] if self.active_sequence else 0
        
        # If new sequence starts with different emotion, add transition frames
        if new_sequence and new_sequence[0][0] != current_emotion_idx:
            # Add transition: current -> neutral -> new_first
            transition = [
                (current_emotion_idx, 10),  # Hold current briefly
                (self._get_cgan_index('neutral'), 15),  # Transition through neutral
            ]
            return transition + new_sequence
        
        return new_sequence

    def get_animator_response(self, kid_detected_cgan_emotion_name):
        """
        Enhanced animator response with therapeutic sequences and smooth transitions.

        Args:
            kid_detected_cgan_emotion_name (str): The detected emotion from the child

        Returns:
            tuple: (cgan_emotion_index_to_display, current_sequence_description_str, has_sequence_changed_bool)
        """
        sequence_changed_this_tick = False
        current_time = time.time()

        # Check if child's emotion has changed
        if kid_detected_cgan_emotion_name != self.current_kid_emotion_name:
            self.current_kid_emotion_name = kid_detected_cgan_emotion_name
            
            # Prevent too rapid sequence changes for stability
            time_since_last_change = current_time - self.last_sequence_change_time
            if time_since_last_change < 2.0:  # Minimum 2 seconds between changes
                # Continue current sequence but note the new emotion for next change
                pass
            else:
                # Select sequence based on mode
                if self.exact_match_mode:
                    # Exact matching mode - mirror the child's emotion
                    chosen_raw_sequence = [(kid_detected_cgan_emotion_name, 60)]
                    new_sequence = [(self._get_cgan_index(name), duration) 
                                  for name, duration in chosen_raw_sequence]
                    sequence_description = f"EXACT MATCH: {kid_detected_cgan_emotion_name}"
                else:
                    # Therapeutic mode - use evidence-based sequences
                    new_sequence = self._select_therapeutic_sequence(kid_detected_cgan_emotion_name)
                    
                    # Get original sequence for description
                    original_sequences = self.therapeutic_sequences.get(
                        kid_detected_cgan_emotion_name, 
                        self.therapeutic_sequences[self.default_sequence_name]
                    )
                    chosen_raw = random.choice(original_sequences)
                    
                    desc_parts = [f"{name}({dur})" for name, dur in chosen_raw]
                    sequence_description = f"THERAPEUTIC - Child:{kid_detected_cgan_emotion_name} -> Response:[{', '.join(desc_parts)}]"

                # Apply smooth transition
                self.active_sequence = self._create_smooth_transition(new_sequence)
                self.sequence_step_index = 0
                self.frames_in_current_step = 0
                self.current_sequence_description = sequence_description
                self.last_sequence_change_time = current_time
                sequence_changed_this_tick = True

        # Handle sequence completion and looping
        if not self.active_sequence or self.sequence_step_index >= len(self.active_sequence):
            # Default to neutral holding pattern when sequence completes
            default_sequence = [(self._get_cgan_index('neutral'), 30), 
                              (self._get_cgan_index('happy'), 15),
                              (self._get_cgan_index('neutral'), 30)]
            self.active_sequence = default_sequence
            self.sequence_step_index = 0
            self.frames_in_current_step = 0
            self.current_sequence_description = "Holding pattern: neutral-happy-neutral"

        # Get current emotion and duration from the active sequence
        current_emotion_index, current_duration_frames = self.active_sequence[self.sequence_step_index]

        # Advance frame counter
        self.frames_in_current_step += 1
        
        # Check if current step is complete
        if self.frames_in_current_step >= current_duration_frames:
            self.sequence_step_index += 1
            self.frames_in_current_step = 0

        # Handle sequence completion
        if self.sequence_step_index >= len(self.active_sequence):
            # Loop back to maintain continuity
            self.sequence_step_index = max(0, len(self.active_sequence) - 1)
            self.frames_in_current_step = 0

        return current_emotion_index, self.current_sequence_description, sequence_changed_this_tick

    def get_status_info(self):
        """Get current status information for debugging"""
        return {
            'mode': 'Exact Match' if self.exact_match_mode else 'Therapeutic',
            'current_child_emotion': self.current_kid_emotion_name,
            'sequence_step': f"{self.sequence_step_index + 1}/{len(self.active_sequence)}",
            'frames_in_step': f"{self.frames_in_current_step}",
            'sequence_description': self.current_sequence_description
        }

if __name__ == '__main__':
    # Testing with both modes
    CGAN_EMOTIONS_TEST = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sadness', 'surprise']
    
    print("Testing Enhanced PlayfulResponder with Therapeutic Sequences")
    print("=" * 60)
    
    # Test therapeutic mode
    responder = PlayfulResponder(CGAN_EMOTIONS_TEST, exact_match_mode=False)
    
    print("\nTEST 1: Therapeutic Mode")
    print("-" * 30)
    
    test_emotions = ['anger', 'sadness', 'fear', 'happy', 'neutral']
    
    for emotion in test_emotions:
        print(f"\nChild shows: {emotion}")
        for i in range(20):  # Show 20 frames of response
            animator_idx, desc, changed = responder.get_animator_response(emotion)
            animator_name = CGAN_EMOTIONS_TEST[animator_idx]
            
            if changed or i == 0:
                print(f"  New sequence: {desc}")
            
            if i % 5 == 0:  # Print every 5th frame
                print(f"    Frame {i+1}: {animator_name}")
        
        time.sleep(0.5)  # Brief pause between emotions
    
    print(f"\n" + "-" * 30)
    print("TEST 2: Exact Match Mode")
    print("-" * 30)
    
    # Test exact match mode
    responder.set_exact_match_mode(True)
    
    for emotion in test_emotions:
        animator_idx, desc, changed = responder.get_animator_response(emotion)
        animator_name = CGAN_EMOTIONS_TEST[animator_idx]
        print(f"Child: {emotion} -> Animator: {animator_name} ({desc})")
    
    print("\nTesting completed.")