package com.pebble.tts;

import android.os.Bundle;
import android.view.MotionEvent;
import android.view.View;
import android.view.ViewConfiguration;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        // Must happen before super.onCreate(): BridgeActivity builds the
        // bridge (and its plugin list) inside super.onCreate(), and this
        // plugin is what writes the exported WAV into public Downloads.
        registerPlugin(DownloadsPlugin.class);

        super.onCreate(savedInstanceState);

        if (bridge == null || bridge.getWebView() == null) {
            return;
        }

        View webView = bridge.getWebView();

        // Disables native Android overscroll / rubber-band bounce
        webView.setOverScrollMode(View.OVER_SCROLL_NEVER);

        // Horizontal drags must never pan the viewport. The CSS already removes
        // every horizontal overflow, but WebView still translates the visual
        // viewport on a sideways swipe, so the gesture is filtered natively
        // here: vertical gestures pass straight through, lateral ones are
        // cancelled for the WebView and swallowed for the rest of the gesture.
        webView.setOnTouchListener(new HorizontalLockTouchListener(webView));

        // Belt and braces: whatever the source of a lateral offset (gesture,
        // fling, programmatic scroll), the WebView's scroll position is pinned
        // to X = 0. Vertical scroll values are passed through untouched.
        webView.setOnScrollChangeListener((v, scrollX, scrollY, oldScrollX, oldScrollY) -> {
            if (scrollX != 0) {
                v.scrollTo(0, scrollY);
            }
        });
    }

    /**
     * Consumes lateral drags before they reach the WebView's scroll machinery
     * while leaving vertical scrolling untouched.
     *
     * <p>ACTION_DOWN is never consumed, so taps, long-presses, focus changes and
     * vertical flings behave exactly as before. The first ACTION_MOVE that is
     * clearly horizontal locks the gesture for its whole lifetime: the WebView
     * receives a synthetic ACTION_CANCEL (which aborts a scroll it may have
     * already started) and every following event, including ACTION_UP, is
     * swallowed so no synthetic click fires at the end of the swipe.
     */
    private static final class HorizontalLockTouchListener implements View.OnTouchListener {

        /** Touch slop in pixels (already density-scaled), typically 8-24px. */
        private final int touchSlop;

        private float startX;
        private float startY;
        private boolean horizontalLocked;

        HorizontalLockTouchListener(View view) {
            this.touchSlop = ViewConfiguration.get(view.getContext()).getScaledTouchSlop();
        }

        @Override
        public boolean onTouch(View v, MotionEvent event) {
            switch (event.getActionMasked()) {
                case MotionEvent.ACTION_DOWN:
                    startX = event.getX();
                    startY = event.getY();
                    horizontalLocked = false;
                    // Let the WebView start the gesture (tap, long-press, vertical scroll).
                    return false;

                case MotionEvent.ACTION_MOVE: {
                    if (horizontalLocked) {
                        return true;
                    }
                    float diffX = Math.abs(event.getX() - startX);
                    float diffY = Math.abs(event.getY() - startY);
                    // Lateral only: it has to beat the slop AND clearly beat the
                    // vertical delta, so normal vertical scrolling never locks.
                    if (diffX > touchSlop && diffX > diffY) {
                        horizontalLocked = true;
                        cancelWebViewGesture(v, event);
                        return true;
                    }
                    return false;
                }

                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL: {
                    boolean wasLocked = horizontalLocked;
                    horizontalLocked = false;
                    // Swallow the end of a blocked swipe so it cannot land as a click.
                    return wasLocked;
                }

                default:
                    return horizontalLocked;
            }
        }

        /**
         * Delivers ACTION_CANCEL straight to the WebView's own touch handling.
         * Calling onTouchEvent (instead of dispatchTouchEvent) bypasses this
         * listener, so the cancel is not swallowed before it reaches the WebView.
         */
        private static void cancelWebViewGesture(View v, MotionEvent event) {
            MotionEvent cancel = MotionEvent.obtain(event);
            cancel.setAction(MotionEvent.ACTION_CANCEL);
            v.onTouchEvent(cancel);
            cancel.recycle();
        }
    }
}
