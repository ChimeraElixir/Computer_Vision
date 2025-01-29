clone = frame.copy()
        cv2.putText(
            clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
        )

        # Loop through all points in the facial landmark region
        for x, y in shape[i:j]:
            cv2.circle(clone, (x, y), 1, (0, 0, 255), -1)

        # Display the frame
        cv2.imshow("Frame", clone)